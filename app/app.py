import requests
import time
from shiny import render, reactive, App, ui
from shiny.types import ImgData
import os
from PIL import Image
import io
import cv2
import numpy as np
import base64

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file(
            id="image",
            label="Satellite Image"
        ),
        ui.input_task_button(
            id="submit",
            label="Submit"
        )
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Vision Model Segmentation Output", class_="d-flex align-items-center"),
            ui.output_image("segmentation_image"),
            ui.card_footer("Segmentation Output"),
            full_screen=True,
            class_="d-flex align-items-center"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Street Model Price Output", class_="d-flex align-items-center"),
                ui.output_ui("price"),
                ui.card_footer("Mask Model Prediction"),
                class_="d-flex align-items-center"),
            ui.card(
                ui.card_header("Street Model Coef Output", class_="d-flex align-items-center"),
                ui.output_ui("coef"),
                ui.card_footer("Mask Model Prediction"),
                class_="d-flex align-items-center")
        ),
        col_widths={"sm": 12, "md": 12, "lg": 12},
        fillable=True),
    ui.include_css("./styles.css"),
    title="Real Estate Satellite",
    fillable=True,
)

def server(input, output, session):
    @reactive.effect
    @reactive.event(input.submit)
    def submit_request():
        if input.image.get() != None:
            image = Image.open(input.image.get()[0]['datapath'])

            image_stream = io.BytesIO()
            image.save(image_stream, format=image.format)
            image_stream.seek(0)

            files = {'image': (f'image.{image.format}', image_stream, f'image/{image.format}')}

            # Segmentation Model
            try:
                response = requests.post(
                    "http://23.240.69.246:4000/satellite_predict",
                    files=files,
                )

                if response.status_code == 200:
                    response_data = response.json()
                    result_image = response_data.get("segmentation_image")
                    result_pred = response_data.get("prediction")
                    street_coeff = response_data.get("coef")

                    img_bytes = base64.b64decode(result_image)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    cv2.imwrite("./segment.png", image)

                    @output
                    @render.image
                    def segmentation_image():
                        return {"src": "./segment.png", "width": "100%", "height": "100%"}
                    
                    @output
                    @render.ui
                    def price():
                        return ui.h1(result_pred)
                    
                    @output
                    @render.ui
                    def coef():
                        return ui.h1(f"{street_coeff:.2f}")

            except Exception as e:
                print("Error in connecting to AVM endpoint:", str(e))

    @output
    @render.image
    def segmentation_image():
        if input.image.get() != None:
            return {"src": input.image.get()[0]['datapath'], "height": "100%", "width" :"100%"}

        return {"src": "./placeholder.png", "height": "100%", "width" :"100%"}
    
    @output
    @render.ui
    def price():
        return ui.p("Enter an image for price prediction")
    
    @output
    @render.ui
    def coef():
        return ui.p("Enter an image for price prediction")

app = App(app_ui, server)
