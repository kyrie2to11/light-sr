import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    # 1. build model and load ckpt
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

    # 2. Export the model with torch.export
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_img = torch.rand(1, 3, 210, 348).to(device)
    exported_model = capture_pre_autograd_graph(_model.model, dummy_img)
    __import__('ipdb').set_trace()
    # 3. Import the backend specific quantizer and configure how to quantize the model
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True))

    # 4. Prepare the model for quantization-aware training
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    
    print(prepared_model)
    __import__('ipdb').set_trace()
    # Manually swap "torch.ops.aten._native_batch_norm_legit" to "torch.ops.aten.cudnn_batch_norm" if needed
    for n in prepared_model.graph.nodes:
        if n.target == torch.ops.aten._native_batch_norm_legit.default:
            n.target = torch.ops.aten.cudnn_batch_norm.default
    prepared_model.recompile()

    # 5. Training loop and save ckpts
    t = Trainer(args, loader, prepared_model, _loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()

    # 7. Convert the trained model to a quantized model
    quantized_model = convert_pt2e(prepared_model)

    # move certain ops like dropout to eval mode, equivalent to `m.eval()`
    torch.ao.quantization.move_exported_model_to_eval(quantized_model)

    print(quantized_model)

if __name__ == "__main__":
    main()