"""Package the 8B paired pilot only after the real GPU feasibility gate passes."""
import json
from pathlib import Path


def replace(source, old, new):
    assert old in source, old
    return source.replace(old, new)


def build():
    evidence = json.loads(Path('artifacts/improvement-v24/kaggle-output/result.json').read_text())
    assert evidence['completed'] and evidence['reload_check'] == 'passed'
    assert evidence['changed_tensors'] > 0 and evidence['skipped_updates'] == 0
    assert evidence['projected_5580_microsteps_seconds'] + 1800 <= 24300
    source = Path('scripts/train_adapter_v18.py').read_text()
    source = replace(source, 'from peft import LoraConfig, get_peft_model',
                     'from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training')
    source = replace(source, 'from transformers import AutoModel',
                     'from transformers import AutoModel, BitsAndBytesConfig')
    source = replace(source,
        'backbone = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16,\n'
        '                                            local_files_only=True, attn_implementation="sdpa")',
        'quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",\n'
        '            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)\n'
        '        backbone = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16,\n'
        '            quantization_config=quant, device_map={"":0},\n'
        '            local_files_only=True, attn_implementation="sdpa")\n'
        '        assert backbone.config.hidden_size == 4096 and backbone.config.num_hidden_layers == 36\n'
        '        backbone = prepare_model_for_kbit_training(backbone,\n'
        '            gradient_checkpointing_kwargs={"use_reentrant":False})')
    source = replace(source, 'model = AttributionModel(model_dir, plan).cuda()',
        'model = AttributionModel(model_dir, plan)\n'
        '    model.norm.cuda(); model.head.cuda(); model.projection.cuda()')
    source = replace(source, 'scaler = torch.amp.GradScaler("cuda")',
                     'scaler = torch.amp.GradScaler("cuda", init_scale=1024)')
    compile(source, 'v25-training-worker', 'exec')
    Path('scripts/train_adapter_v25.py').write_text(source)
    readout = Path('scripts/readout_v19.py').read_text()
    readout = replace(readout, 'from transformers import AutoModel, AutoTokenizer',
                      'from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig')
    readout = replace(readout, 'from peft import PeftModel',
                      'from peft import PeftModel, prepare_model_for_kbit_training')
    readout = replace(readout,
        'model = AutoModel.from_pretrained(plan["model_dir"], torch_dtype=torch.float16,\n'
        '                                         attn_implementation="sdpa", local_files_only=True)',
        'quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",\n'
        '            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)\n'
        '        model = AutoModel.from_pretrained(plan["model_dir"], torch_dtype=torch.float16,\n'
        '            quantization_config=quant, device_map={"":0},\n'
        '            attn_implementation="sdpa", local_files_only=True)\n'
        '        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)')
    readout = replace(readout, 'model = model.cuda().eval()', 'model = model.eval()')
    compile(readout, 'v25-readout-worker', 'exec')
    Path('scripts/readout_v25.py').write_text(readout)
    builder = Path('scripts/build_training_v22.py').read_text()
    builder = builder.replace('v22', 'v25').replace('V22', 'V25')
    builder = replace(builder, "template=Path('scripts/build_training_v18.py').read_text()",
        "template=Path('scripts/build_training_v18.py').read_text()\n"
        "    template=template.replace('scripts/train_adapter_v18.py','scripts/train_adapter_v25.py')\n"
        "    template=template.replace('/4b-base/','/8b-base/')")
    builder = replace(builder, 'scripts/readout_v19.py', 'scripts/readout_v25.py')
    builder = builder.replace('name-that-llm-scaled-training-v25','name-that-llm-qlora-training-v25')
    builder = builder.replace('Name That LLM Scaled Training V25','Name That LLM QLoRA Training V25')
    # Replace the inherited explanatory cell after packaging; no code cells are altered.
    namespace = {'__name__':'v25_packaging'}
    exec(compile(builder,'v25-builder','exec'), namespace)
    namespace['build']()
    import nbformat, hashlib
    path = Path('notebooks/training-v25/training.ipynb')
    nb = nbformat.read(path,4)
    nb.cells[0] = nbformat.v4.new_markdown_cell(
        '# V25: 8B QLoRA paired attribution pilot\n\n'
        'Matched CE and CE+SupCon, 18,000 external passages and two host epochs. '
        'Qwen3-8B NF4 with all-linear LoRA; fixed layers12/24/36 readout. '
        'Changes from V22 include model size, quantization and adapter coverage together. '
        'No submission. LoRA: Hu et al. 2106.09685; SupCon: Khosla et al. 2004.11362; '
        'QLoRA: Dettmers et al. 2305.14314 and github.com/artidoro/qlora. '
        'Public pool credited to Nishant Dahal and original contributors, CC BY-SA 4.0.')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'v25-notebook', 'exec')
    nbformat.validate(nb); nbformat.write(nb,path)
    frozen = Path('artifacts/improvement-v25/frozen-plan.json')
    plan = json.loads(frozen.read_text())
    plan['notebook_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan['feasibility_evidence'] = evidence
    frozen.write_text(json.dumps(plan,indent=2))


if __name__ == '__main__':
    build()
