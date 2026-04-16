"""smer_visual - SMER Visual interpretable image classification with LFG integration."""

from .smer import (
    # Original SMER functions (word descriptions approach)
    image_descriptions,
    embed_descriptions,
    aggregate_embeddings,
    classify_lr,
    compute_aopc,
    build_custom_predict,
    plot_aopc,
    plot_important_words,
    save_bounding_box_images,
    # LFG Feature Discovery & Generation
    discover_features,
    generate_features,
    load_lfg_features,
    # SMER on LFG Features (hybrid approach)
    embed_lfg_features,
    classify_lr_lfg,
    compute_aopc_lfg,
    build_custom_predict_lfg,
    plot_aopc_lfg,
    plot_important_features_lfg,
    smer_lfg_pipeline,
    # Unified pipelines
    smer_text_pipeline,
    smer_pipeline,
    # LFG availability flag
    LFG_AVAILABLE,
)

__all__ = [
    # Original SMER functions (word descriptions approach)
    "image_descriptions",
    "embed_descriptions",
    "aggregate_embeddings",
    "classify_lr",
    "compute_aopc",
    "build_custom_predict",
    "plot_aopc",
    "plot_important_words",
    "save_bounding_box_images",
    # LFG Feature Discovery & Generation
    "discover_features",
    "generate_features",
    "load_lfg_features",
    # SMER on LFG Features (hybrid approach)
    "embed_lfg_features",
    "classify_lr_lfg",
    "compute_aopc_lfg",
    "build_custom_predict_lfg",
    "plot_aopc_lfg",
    "plot_important_features_lfg",
    "smer_lfg_pipeline",
    # Standalone LFG Classification (non-SMER)
    "encode_lfg_features",
    "classify_lfg_features",
    "get_lfg_feature_importance",
    "plot_lfg_feature_importance",
    "lfg_pipeline",
    # Unified pipelines
    "smer_text_pipeline",
    "smer_pipeline",
    # LFG availability flag
    "LFG_AVAILABLE",
]

