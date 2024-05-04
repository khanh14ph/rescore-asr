from model import MyModel
torch_model = MyModel()
model.load_state_dict(torch.load("/home4/khanhnd/pbert/best_checkpoint_10_hidden.pt",map_location=torch.device(device)))
torch_model.eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
dummy_input=tokenizer("tôi ko hiểu", padding=True, truncation=True, return_tensors="pt")
# Export the model
torch.onnx.export(
    self.bert, 
    tuple(dummy_input.values()),
    f="torch-model.onnx",  
    input_names=['input_ids', 'attention_mask'], 
    output_names=['logits'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                  'logits': {0: 'batch_size', 1: 'sequence'}}, 
    do_constant_folding=True, 
    opset_version=13, 
)
docker run -it --name slu_basesolver_container --gpus all -v ./:/slu -v {/home4/khanhnd/DATA_DIR}:/slu/data 20011911/soict2023_slu_basesolver
