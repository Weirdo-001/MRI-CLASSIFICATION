import tensorflow as tf
import numpy as np
import matplotlib
from PIL import Image


def get_last_conv_layer(model):
    """Find the best conv layer for Grad-CAM in EfficientNet models."""
    layer_names = [l.name for l in model.layers]

    # Prefer EfficientNet's top_conv — ideal resolution for Grad-CAM
    preferred = ["top_conv", "block7b_project_conv", "block7a_project_conv",
                 "block6d_project_conv", "block6c_project_conv"]
    for name in preferred:
        if name in layer_names:
            print(f"[GradCAM] Using preferred layer: {name}")
            return name

    # Fallback: last layer with 4D output
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 4:
                print(f"[GradCAM] Fallback layer: {layer.name}")
                return layer.name
        except Exception:
            continue
    return None


def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        print(f"[GradCAM] Could not build grad model with '{last_conv_layer_name}': {e}")
        return None

    # KEY FIX: cast to float32 tensor and watch it BEFORE the forward pass.
    # GradientTape only auto-watches tf.Variables. By watching img_tensor explicitly,
    # TF records the full computation graph: input → conv_outputs → predictions → class_channel,
    # which lets us compute d(class_channel)/d(conv_outputs) correctly.
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        model_inputs = [img_tensor] if len(model.inputs) == 1 else img_tensor
        conv_outputs, predictions = grad_model(model_inputs)

        # predictions can be a list — normalise to single float32 tensor
        if isinstance(predictions, (list, tuple)):
            predictions = tf.concat([tf.cast(p, tf.float32) for p in predictions], axis=-1)
        predictions = tf.cast(predictions, tf.float32)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("[GradCAM] Gradients are still None — try a different conv layer.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        print("[GradCAM] Heatmap is all zeros.")
        return None
    return (heatmap / max_val).numpy()


def overlay_gradcam(img, heatmap, alpha=0.55):
    """img: H×W×3 uint8 numpy array. Returns (pure_heatmap_rgb, overlay_rgb)."""
    # Power-normalize: push weak activations to zero so only the hottest zone shows red
    heatmap = np.power(heatmap, 2.0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_uint8 = np.uint8(255 * heatmap)

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]        # (256, 3) float64 0-1
    jet_heatmap = jet_colors[heatmap_uint8]         # (H, W, 3) float64 0-1

    jet_pil = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_pil = jet_pil.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    jet_heatmap_resized = np.array(jet_pil).astype(np.float32)  # 0-255

    img_float = img.astype(np.float32)
    superimposed = jet_heatmap_resized * alpha + img_float * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    pure_heatmap = np.uint8(jet_heatmap_resized)

    return pure_heatmap, superimposed
