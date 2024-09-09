import flet as ft
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import numpy as np
import io

# Загрузка модели YOLO (замените на путь к вашей .pt модели)
model = YOLO("best.pt")

class DrawingApp(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.canvas_image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.canvas_image)
        self.is_drawing = False
        self.last_x, self.last_y = None, None

    def build(self):
        # Кнопка для очистки экрана
        clear_button = ft.ElevatedButton(text="Clear", on_click=self.clear_canvas)
        recognize_button = ft.ElevatedButton(text="Recognize", on_click=self.recognize_image)
        save_button = ft.ElevatedButton(text="Save", on_click=self.save_image)
        self.output_field = ft.Text("Drawing recognition result will appear here.")

        # Канвас для рисования
        self.canvas = ft.Canvas(width=400, height=400, on_pointer_down=self.start_draw,
                                on_pointer_move=self.draw_line, on_pointer_up=self.stop_draw)

        return ft.Row(controls=[
            ft.Column([self.canvas, clear_button], alignment=ft.MainAxisAlignment.CENTER),
            ft.Column([recognize_button, save_button, self.output_field],
                      alignment=ft.MainAxisAlignment.CENTER)
        ])

    def start_draw(self, e: ft.PointerEvent):
        self.is_drawing = True
        self.last_x, self.last_y = e.local_x, e.local_y

    def draw_line(self, e: ft.PointerEvent):
        if self.is_drawing:
            x, y = e.local_x, e.local_y
            self.canvas_image.line([self.last_x, self.last_y, x, y], fill="black", width=5)
            self.last_x, self.last_y = x, y
            self.update()

    def stop_draw(self, e: ft.PointerEvent):
        self.is_drawing = False

    def clear_canvas(self, e):
        self.canvas_image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.canvas_image)
        self.update()

    def recognize_image(self, e):
        # Преобразование изображения в нужный формат для модели
        img_byte_arr = io.BytesIO()
        self.canvas_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_tensor = torch.from_numpy(np.array(self.canvas_image)).float()

        # Использование модели для предсказания
        results = model(img_tensor)

        # Вывод информации о распознавании
        self.output_field.value = str(results)
        self.update()

    def save_image(self, e):
        self.canvas_image.save("drawing.png")
        self.output_field.value = "Image saved as 'drawing.png'."
        self.update()

def main(page: ft.Page):
    page.title = "Drawing Recognition App"
    app = DrawingApp()
    page.add(app)

ft.app(target=main)
