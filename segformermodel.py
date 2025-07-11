from transformers import SegformerForSemanticSegmentation

def get_segformer_model(pretrained_model: str, num_labels=1):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return model