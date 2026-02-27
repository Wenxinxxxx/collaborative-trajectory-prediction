from .lstm_seq2seq import LSTMSeq2Seq
from .social_lstm import SocialLSTM
from .grip_plus import GRIPPlusPlus
from .transformer_pred import TransformerPredictor
from .v2x_graph import V2XGraphPredictor
from .co_mtp import CoMTP
from .enhanced_co_mtp import EnhancedCoMTP

MODEL_REGISTRY = {
    'lstm_seq2seq': LSTMSeq2Seq,
    'social_lstm': SocialLSTM,
    'grip_plus': GRIPPlusPlus,
    'transformer': TransformerPredictor,
    'v2x_graph': V2XGraphPredictor,
    'co_mtp': CoMTP,
    'enhanced_co_mtp': EnhancedCoMTP,
}


def get_model(model_name, config):
    """Factory function to create a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)
