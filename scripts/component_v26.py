"""Prepared 8B NF4 portable readout; launch requires a qualified V25 result.

Retains the V21 portable fit/predict math; loads the backbone as in V25.
Method credit: QLoRA (Dettmers et al., arXiv:2305.14314), Hugging Face PEFT.
"""
import gc
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from scipy.special import log_softmax


class ContrastiveReadout:
    def __init__(self, model_dir, adapter, layers=(12,24,36), max_length=256, batch_size=8):
        self.model_dir, self.adapter = model_dir, adapter
        self.layers, self.max_length, self.batch_size = list(layers), max_length, batch_size

    def features(self, texts):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir,local_files_only=True)
        tokenizer.padding_side='right'
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        tokens=tokenizer(list(texts),truncation=True,max_length=self.max_length-1,
                         add_special_tokens=False)['input_ids']
        tokens=[ids+[tokenizer.eos_token_id] for ids in tokens]
        order=np.argsort([len(ids) for ids in tokens],kind='stable')
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                 bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModel.from_pretrained(self.model_dir,torch_dtype=torch.float16,
            quantization_config=quant,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
        assert model.config.hidden_size==4096 and model.config.num_hidden_layers==36
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model=PeftModel.from_pretrained(model,self.adapter,is_trainable=False).eval()
        features=np.empty((len(texts),len(self.layers),model.config.hidden_size),dtype=np.float32)
        started=time.monotonic()
        with torch.inference_mode():
            for start in range(0,len(texts),self.batch_size):
                ids=order[start:start+self.batch_size]
                batch={k:v.cuda() for k,v in tokenizer.pad({'input_ids':[tokens[i] for i in ids]},
                                                          padding=True,return_tensors='pt').items()}
                with torch.autocast('cuda',dtype=torch.float16):
                    states=model(**batch,output_hidden_states=True,use_cache=False).hidden_states
                mask=batch['attention_mask'].clone();mask[:,0]=0
                for j,layer in enumerate(self.layers):
                    h=states[layer].float()
                    features[ids,j]=((h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).cpu().numpy()
                del states
                if start%400==0:
                    print(json.dumps({'event':'v26_features','completed':min(start+len(ids),len(texts)),
                                      'total':len(texts),'seconds':time.monotonic()-started}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
        assert np.isfinite(features).all()
        return features

    @staticmethod
    def predict(features, path):
        with np.load(path,allow_pickle=False) as state:
            assert features.shape[1:]==(len(state['layers']),state['mean'].shape[-1])
            scores=[]
            for j in range(features.shape[1]):
                X=(features[:,j].astype(np.float64)-state['mean'][j])/state['scale'][j]
                scores.append(log_softmax(X@state['coef'][j].T+state['intercept'][j],axis=1))
            labels=state['labels'].copy()
        scores=np.mean(scores,axis=0)
        assert np.isfinite(scores).all()
        return labels,scores

    def fit(self,features,targets,path,C=0.02,max_iter=1000,seed=1901):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        means,scales,coefs,intercepts,predictions=[],[],[],[],[]
        labels=np.sort(np.unique(targets))
        for j,layer in enumerate(self.layers):
            scaler=StandardScaler()
            X=scaler.fit_transform(features[:,j].astype(np.float64))
            clf=LogisticRegression(C=C,max_iter=max_iter,random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                clf.fit(X,targets)
            assert np.array_equal(labels,clf.classes_)
            means.append(scaler.mean_);scales.append(scaler.scale_)
            coefs.append(clf.coef_);intercepts.append(clf.intercept_)
            predictions.append(clf.predict_log_proba(X))
        np.savez_compressed(path,labels=labels.astype(str),layers=self.layers,mean=np.stack(means),
                            scale=np.stack(scales),coef=np.stack(coefs),intercept=np.stack(intercepts))
        recovered_labels,recovered_scores=self.predict(features,path)
        assert np.array_equal(recovered_labels,labels)
        np.testing.assert_allclose(recovered_scores,np.mean(predictions,axis=0),atol=1e-8,rtol=1e-8)
        return {'fit_rows':len(targets),'layers':self.layers,'labels':labels.tolist(),
                'portable_readout_check':'passed','C':C,'dtype':'float64 for scaler and LR'}


def main():
    import hashlib
    import pandas as pd
    plan=json.loads(Path('plan.json').read_text())
    assert plan['experiment']=='improvement-v26' and plan['pilot_qualification']['verified']
    frame=pd.read_csv(plan['host_path'],dtype={'id':str})
    component=ContrastiveReadout(plan['model_dir'],'full/adapter',layers=plan['readout']['layers'],
                                max_length=plan['training_parameters']['max_length'],
                                batch_size=plan['readout']['batch_size'])
    features=component.features(frame.text.tolist())
    np.save('full/train_features.npy',features)
    result=component.fit(features,frame.model.to_numpy(),'full/readout.npz',
                          C=plan['readout']['C'],max_iter=plan['readout']['max_iter'],seed=plan['readout']['seed'])
    result['artifacts_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path('full/adapter/adapter_model.safetensors'),Path('full/readout.npz'),Path('full/train_features.npy')]}
    Path('full/readout_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
