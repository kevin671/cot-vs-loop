from .tmlt import TimeModulatedLoopedTF
from .transformer import CT, GPT, CTConfig, GPTConfig, LoopedTF, LoopedTFConfig


def build_model(args, task):
    tf_config = dict(
        block_size=task.config["max_length"],
        vocab_size=task.config["vocab_size"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )

    if args.model in ["Looped", "TMLT"]:
        model_config = LoopedTFConfig(
            **tf_config,
            n_loop=args.n_loop,
            is_causal=args.is_causal,
            # use_rope=args.use_rope,
        )
        if args.model == "Looped":
            model = LoopedTF(model_config)
        else:
            model = TimeModulatedLoopedTF(model_config)

    elif args.model == "CT":
        tf_config["block_size"] += args.n_step + 1
        model_config = CTConfig(**tf_config, n_step=args.n_step)
        model = CT(model_config)

    else:
        model_config = GPTConfig(**tf_config)
        model = GPT(model_config)

    return model
