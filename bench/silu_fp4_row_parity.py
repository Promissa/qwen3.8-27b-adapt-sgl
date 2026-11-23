#!/usr/bin/env python3
"""Native fused SwiGLU/FP4 rows versus its own sequential-row control."""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
from common import ROOT, now, write_json


def run(output):
    import torch
    from safetensors import safe_open
    from sglang.srt.batch_invariant_ops import set_batch_invariant_mode
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize, initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp4LinearMethod
    from sglang.srt.models.utils import silu_and_mul_fp4_quant
    from sglang.srt.runtime_context import get_context, get_exec

    checkpoint=ROOT/'weights/target'
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    config=json.loads((checkpoint/'config.json').read_text())['quantization_config']
    result={'at':now(),'status':'running','cases':[], 'modules':[],
        'scope':'Actual down-projection weights/scales with synthetic BF16 gate/up inputs. Isolates fused activation/quantization row batching, not full-model equality or speed.',
        'target_revision':json.loads((checkpoint/'revision.json').read_text()),
        'device':torch.cuda.get_device_name(),'torch_version':torch.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fused_source_sha256':hashlib.sha256(Path(inspect.getsourcefile(silu_and_mul_fp4_quant)).read_bytes()).hexdigest()}
    write_json(output,result)

    def tensor(name):
        with safe_open(checkpoint/index[name],framework='pt',device='cpu') as handle:
            return handle.get_tensor(name)

    def scales(value,rows,columns):
        rp,cp=(rows+127)//128*128,(columns+3)//4*4
        return value.view(torch.uint8).reshape(1,rp//128,cp//4,32,4,4).permute(
            0,1,4,3,2,5).contiguous().reshape(rp,cp)[:rows,:columns]

    def difference(left,right):
        if left.shape!=right.shape:
            raise ValueError(f'Unexpected output shapes: {left.shape}/{right.shape}')
        finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if not finite:raise ValueError('Nonfinite isolated operator output')
        return {'exact':torch.equal(left,right),'different_elements':int((left!=right).sum().item()),
                'elements':left.numel(),'maximum_absolute_difference':float((left.float()-right.float()).abs().max().item())}

    started=time.monotonic()
    try:
        with get_context().override_server_args(), get_exec().kernel.override(fp4_gemm_runner_backend='flashinfer_cutlass'):
            initialize_fp4_gemm_config()
            quantization=ModelOptFp4Config.from_config(config)
            for layer_id in (0,3,30,62):
                prefix=f'model.language_model.layers.{layer_id}.mlp.down_proj'
                weight=tensor(prefix+'.weight');n,k=weight.shape[0],weight.shape[1]*2
                with torch.device('cuda'):
                    down=torch.nn.Module();method=ModelOptFp4LinearMethod(quantization)
                    assert method.quant_mode=='w4a4'
                    method.create_weights(down,input_size_per_partition=k,output_partition_sizes=[n],
                        input_size=k,output_size=n,params_dtype=torch.bfloat16)
                    with torch.no_grad():
                        for suffix in ('weight','weight_scale','input_scale','weight_scale_2'):
                            value=weight if suffix=='weight' else tensor(prefix+'.'+suffix)
                            getattr(down,suffix).copy_(value.reshape_as(getattr(down,suffix)))
                    method.process_weights_after_loading(down)
                del weight,value
                down._accepts_prequantized_fp4=True
                mlp=SimpleNamespace(down_proj=down,_down_input_scale_inv_1d=None,_masked_m_cache={})
                activation=SiluAndMul()
                result['modules'].append({'layer':layer_id,'checkpoint_prefix':prefix,'n':n,'k':k,'quant_mode':method.quant_mode})
                torch.manual_seed(1701+layer_id)
                source=torch.randn(128,2*k,device='cuda',dtype=torch.bfloat16)
                for magnitude in (.125,1.):
                    x=source*magnitude
                    for invariant in (False,True):
                        with torch.inference_mode(),set_batch_invariant_mode(invariant):
                            sequential=[silu_and_mul_fp4_quant(mlp,x[i:i+1]) for i in range(128)]
                            seq_q=torch.cat([v[0] for v in sequential])
                            seq_sf=torch.cat([scales(v[1],1,k//16) for v in sequential])
                            seq_out=torch.cat([method.apply(down,v) for v in sequential])
                            for rows in (1,2,4,8,32,64,128):
                                q,sf=silu_and_mul_fp4_quant(mlp,x[:rows])
                                logical=scales(sf,rows,k//16)
                                fused=method.apply(down,(q,sf))
                                separate_activation=activation(x[:rows])
                                separate_q,separate_sf=fp4_quantize(separate_activation,down.input_scale_inv)
                                separate_logical=scales(separate_sf,rows,k//16)
                                separate_out=method.apply(down,separate_activation)
                                result['cases'].append({'layer':layer_id,'rows':rows,'magnitude':magnitude,
                                    'aten_batch_invariant':invariant,
                                    'fused_vs_sequential_packed':torch.equal(q,seq_q[:rows]),
                                    'fused_vs_sequential_scales':torch.equal(logical,seq_sf[:rows]),
                                    'fused_vs_sequential_output':difference(fused,seq_out[:rows]),
                                    'fused_vs_separate_packed':torch.equal(q,separate_q),
                                    'fused_vs_separate_scales':torch.equal(logical,separate_logical),
                                    'fused_vs_separate_output':difference(fused,separate_out)})
                        write_json(output,result)
                del down,method,mlp,source,x,sequential,seq_q,seq_sf,seq_out,q,sf,fused,separate_activation,separate_q,separate_sf,separate_out
                gc.collect();torch.cuda.empty_cache()
            expected={(layer,rows,mag,inv) for layer in (0,3,30,62) for rows in (1,2,4,8,32,64,128) for mag in (.125,1.) for inv in (False,True)}
            actual={(c['layer'],c['rows'],c['magnitude'],c['aten_batch_invariant']) for c in result['cases']}
            assert actual==expected and len(result['cases'])==len(expected)==112
            result.update(status='completed',cases_count=len(result['cases']),
                exact_fused_row_output_cases=sum(c['fused_vs_sequential_output']['exact'] for c in result['cases']),
                exact_fused_row_packed_cases=sum(c['fused_vs_sequential_packed'] for c in result['cases']),
                exact_fused_row_scale_cases=sum(c['fused_vs_sequential_scales'] for c in result['cases']),
                elapsed_s=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output,result)
    print(json.dumps({k:result[k] for k in ('status','cases_count','exact_fused_row_output_cases','exact_fused_row_packed_cases','exact_fused_row_scale_cases','elapsed_s')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new evidence file')
    run(args.output)
