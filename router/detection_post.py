####################################### IMPORT #################################
import json
from loguru import logger
import sys
from starlette.responses import Response

from fastapi import FastAPI, APIRouter, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

import io
from PIL import Image
from PIL import ImageDraw, ImageFont

from app import get_image_from_bytes
from app import detect_sample_model
from app import segment_sample_model
from app import add_bboxs_on_img
from app import get_bytes_from_image
from app import censor_objects

router = APIRouter(prefix='/detection', tags=['Detection'])


######################### MAIN Func #################################


@router.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    **Object Detection from an image.**

    **Args:**
        - **file (bytes)**: The image file in bytes format.
    **Returns:**
        - **dict**: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)

    # Check if no objects are detected
    if len(predict) == 0:
        logger.info("No objects detected.")
        return {"message": "No objects detected."}

    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values

    result = {
        'detect_objects_names': ', '.join(objects),
        'detect_objects': json.loads(detect_res.to_json(orient='records'))
    }
    return result

@router.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    **Object Detection from an image plot bbox on image**

    **Args:**
        - **file (bytes)** The image file in bytes format.
    **Returns:**
        - **Image** Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_sample_model(input_image)

    # Check if no objects are detected
    if len(predict) == 0:
        logger.info("NO OBJECTS DETECTED.")

        #Add annotated text to image stating no objects detected
        draw = ImageDraw.Draw(input_image)
        font = ImageFont.truetype("arial.ttf", 15)  # Adjust with your font path and size

        text = "NO OBJECTS DETECTED."
        x, y = 20, 20  # Text position
        fill_color = (255, 165, 0)  # Orange color for the text
        outline_color = 'black'  # Color for the outline

        # Draw outline by offsetting the text position in all directions
        for adj in range(-2, 3):  # Adjust range for thicker/thinner outline
            # Move right and left
            draw.text((x+adj, y), text, font=font, fill=outline_color)
            # Move up and down
            draw.text((x, y+adj), text, font=font, fill=outline_color)
            # Diagonal moves
            draw.text((x+adj, y+adj), text, font=font, fill=outline_color)
            draw.text((x-adj, y-adj), text, font=font, fill=outline_color)

        # Draw the main text
        draw.text((x, y), text, font=font, fill=fill_color)

        # return annotated image in bytes format
        return StreamingResponse(content=get_bytes_from_image(input_image), media_type="image/jpeg")
    else:
        final_image = add_bboxs_on_img(image = input_image, predict = predict)
        # return image in bytes format
        return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")

    

@router.post("/img_object_detection_to_censored_img")
def img_object_detection_to_censored_img(file: bytes = File(...)):
    """
    **Object Detection from an image plot bbox on image**

    **Args:**
        - **file (bytes)** The image file in bytes format.
    **Returns:**
        - **Image** Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_sample_model(input_image)

    # Check if no objects are detected
    if len(predict) == 0:
        logger.info("NO OBJECTS DETECTED.")

        #Add annotated text to image stating no objects detected
        draw = ImageDraw.Draw(input_image)
        font = ImageFont.truetype("arial.ttf", 15)  # Adjust with your font path and size

        text = "NO OBJECTS DETECTED."
        x, y = 20, 20  # Text position
        fill_color = (255, 165, 0)  # Orange color for the text
        outline_color = 'black'  # Color for the outline

        # Draw outline by offsetting the text position in all directions
        for adj in range(-2, 3):  # Adjust range for thicker/thinner outline
            # Move right and left
            draw.text((x+adj, y), text, font=font, fill=outline_color)
            # Move up and down
            draw.text((x, y+adj), text, font=font, fill=outline_color)
            # Diagonal moves
            draw.text((x+adj, y+adj), text, font=font, fill=outline_color)
            draw.text((x-adj, y-adj), text, font=font, fill=outline_color)

        # Draw the main text
        draw.text((x, y), text, font=font, fill=fill_color)

        # return annotated image in bytes format
        return StreamingResponse(content=get_bytes_from_image(input_image), media_type="image/jpeg")
    else:
        censored_image = censor_objects(input_image, predict, method='blur')
        # return image in bytes format
        return StreamingResponse(content=get_bytes_from_image(censored_image), media_type="image/jpeg")



# @router.post("/img_object_segmentation_to_img",  tags=['Object Segmentation'])
# def img_object_segmentation_to_img(file: bytes = File(...)):
#     """
#     **Object Segmentation from an image plot bbox on image**

#     **Args:**
#        - **file (bytes)**: The image file in bytes format.
#     **Returns:**
#        - **Image**: Image in bytes with bbox annotations.
#     """
#     # get image from bytes
#     input_image = get_image_from_bytes(file)

#     # model predict
#     predict = segment_sample_model(input_image)

#     # add bbox on image
#     final_image = add_bboxs_on_img(image = input_image, predict = predict)

#     # return image in bytes format
#     #return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")
#     predict.render()  # updates results.imgs with boxes and labels
#     for img in predict.imgs:
#         bytes_io = io.BytesIO()
#         img_base64 = Image.fromarray(img)
#         img_base64.save(bytes_io, format="jpeg")
#     return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
#     # return Image(content=get_bytes_from_image(final_image), media_type="image/jpeg", height=600)


