from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils

# Caminhos
MODEL_PATH = "model/model.tflite"  # seu modelo treinado
LABEL_FILE = "model/labels.txt"  # suas labels
MODEL_WITH_METADATA = "model/model_metadata.tflite"

# Cria o writer
writer = image_classifier.MetadataWriter.create_for_inference(
    writer_utils.load_file(MODEL_PATH),
    input_norm_mean=[0.0],    # normalização do modelo
    input_norm_std=[1.0],     # idem
    label_file_paths=[LABEL_FILE]
)

# Salva o modelo com metadata embutida
writer_utils.save_file(writer.populate(), MODEL_WITH_METADATA)

print("Novo modelo salvo em:", MODEL_WITH_METADATA)