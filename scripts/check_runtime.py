"""GPU 实验前检查 OFT 注意力语义；普通 Transformers 同版本号也不算通过。"""
import hashlib
import inspect
import json
from pathlib import Path


def main():
    import torch
    import transformers
    from transformers.models.llama import modeling_llama as llama
    source = Path(inspect.getsourcefile(llama))
    expected = json.loads((Path(__file__).resolve().parents[1] / "configs/versions.json").read_text(encoding="utf-8"))["transformers_oft"]
    actual = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != expected["modeling_llama_lf_sha256"]:
        raise RuntimeError("OFT modeling_llama.py 与固定分支不符，禁止用普通 Transformers 继续实验")
    config = transformers.LlamaConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=128, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = llama.LlamaModel(config).eval()
    torch.manual_seed(7)
    x = torch.randn(1, 6, 32)
    y = x.clone()
    y[:, -1, :] += torch.randn(32) * 2
    with torch.no_grad():
        a = model(inputs_embeds=x, attention_mask=torch.ones(1, 6), use_cache=False).last_hidden_state
        b = model(inputs_embeds=y, attention_mask=torch.ones(1, 6), use_cache=False).last_hidden_state
    if torch.allclose(a[:, 0], b[:, 0], atol=1e-7, rtol=1e-7):
        raise RuntimeError("后方 token 未影响前方 token：检测到因果掩码，OFT 双向语义未生效")
    print(json.dumps({"status": "PASS", "transformers": transformers.__version__,
          "modeling_llama_sha256": actual, "backend": "sdpa", "bidirectional_test": True}))


if __name__ == "__main__":
    main()
