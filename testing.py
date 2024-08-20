from pathlib import Path
import traceback
import uuid
import os
from unstructured.partition.pdf import partition_pdf
# Define paths
pdf_directory = Path("./data")
image_path = Path("./images1")
image_path.mkdir(exist_ok=True, parents=True)

# Dictionary to store image metadata
image_metadata_dict = {}

# Limit the number of images downloaded per PDF
MAX_IMAGES_PER_PDF = 15

# Iterate over each PDF file in the data folder
for pdf_file in pdf_directory.glob("*.pdf"):
    images_per_pdf = 0
    print(f"Processing: {pdf_file}")

    # Extract images from the PDF
    try:
        raw_pdf_elements = partition_pdf(
            filename=str(pdf_file),
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=image_path,
        )

        # Loop through the elements
        for element in raw_pdf_elements:
            # Print the type and metadata of each element
            print(f"Element type: {type(element)}")
            print(f"Metadata for element: {element.metadata.__dict__}")

            # Check if the element contains an image by looking for image-related attributes in the metadata
            if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
                print(f"Image base64 attribute found and is not empty.")
                image_uuid = str(uuid.uuid4())  # Generate a new UUID for each image
                image_file_name = f"{pdf_file.stem}_{image_uuid}.jpg"

                # Convert base64 to image and save
                image_data = element.metadata.image_base64
                with open(os.path.join(image_path, image_file_name), "wb") as img_file:
                    img_file.write(base64.b64decode(image_data))
                
                # Retrieve emphasized text content if available
                emphasized_text_content = getattr(element.metadata, 'emphasized_text_contents', None)

                # Save image metadata including emphasized text content if available
                image_metadata_dict[image_uuid] = {
                    "filename": image_file_name,
                    "img_path": os.path.join(image_path, image_file_name),
                    "emphasized_text_content": emphasized_text_content,
                }

                images_per_pdf += 1

                # Limit the number of images downloaded per PDF
                if images_per_pdf >= MAX_IMAGES_PER_PDF:
                    break
            else:
                print("Image base64 attribute not found or is empty.")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        traceback.print_exc()

# Print the number of items in image_metadata_dict
print(f"Number of items in image_metadata_dict: {len(image_metadata_dict)}")