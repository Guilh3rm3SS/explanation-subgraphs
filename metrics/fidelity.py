from torch_geometric.explain.metric import fidelity, characterization_score, unfaithfulness

def get_fidelity_metrics(explanation, explainer, is_Synthetic=False):
    fid_pos, fid_neg = fidelity(explainer, explanation)
    unfaith = unfaithfulness(explainer, explanation)
    
    try:
        char_score = characterization_score(fid_pos, fid_neg)
    except (ZeroDivisionError, ValueError):
        # ValueError can also happen if metrics are invalid
        char_score = 0.0
    
    return {
        "fidelity+": fid_pos,
        "fidelity-": fid_neg,
        "unfaithfulness": unfaith,
        "characterization_score": char_score
    }
