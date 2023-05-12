from utils.utils import Config

def model_config():
    config = Config({
        # ELIC
        "N": 192,
        "M": 320,
        "slice_num": 5,
        "context_window": 5,
        "slice_ch": [16, 16, 32, 64, 192],
        "quant": "ste",
        
        # rd lambda
        "lambda_char": 2e-6,
        "lambda_lpips": 1,
        "lambda_style": 1e2,
        "lambda_face": 0,
        "lambda_gan": 1,
        "lambda_rate": 0.3,

    })

    return config
