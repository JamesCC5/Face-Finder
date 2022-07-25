import tkinter
import tkinter as tk
import tkinter.messagebox
import customtkinter
import face_recognition
import numpy as np
from tkinter import filedialog
from glob import glob
import shutil
from PIL import UnidentifiedImageError

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    WIDTH = 700
    HEIGHT = 480

    def __init__(self):
        super().__init__()

        self.title("Face Finder")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        self.iconphoto(True, tk.PhotoImage(file='assets/facedetect.png'))
        self.frame_main()

    def frame_main(self):
        # ============ create down frame ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(0, weight=1)  # empty column as spacing
        self.grid_rowconfigure(0, weight=1)  # empty row as spacing

        self.frame_down = customtkinter.CTkFrame(master=self,
                                                 height=120,
                                                 corner_radius=0)
        self.frame_down.grid(row=1, column=0, sticky="nswe")

        # ============ down frame components ============

        # configure grid layout (3x1)
        self.frame_down.grid_columnconfigure(1, weight=1)  # empty row as spacing

        # Things happen in the downside
        self.label_left = customtkinter.CTkLabel(master=self.frame_down,
                                                 text="Face Finder",
                                                 text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_left.grid(row=1, column=0, pady=10, padx=10)

        self.button_process = customtkinter.CTkButton(master=self.frame_down,
                                                      text="Process",
                                                      text_font=("Roboto Light", -14),  # font name and size in px
                                                      command=self.button_process)
        self.button_process.grid(row=1, column=2, pady=10, padx=20)

        # ============ create up frame ============

        self.frame_up = customtkinter.CTkFrame(master=self)
        self.frame_up.grid(row=0, column=0, sticky="nswe", padx=20, pady=20)

        # ============ up frame components ============

        self.frame_up.columnconfigure((1, 5, 6, 7, 11, 12, 13, 17, 18), weight=1)
        self.frame_up.rowconfigure(0, minsize=100)

        # Settings button
        self.button_settings = customtkinter.CTkButton(master=self.frame_up,
                                                       text="≡",
                                                       text_font=("Roboto Medium", -50),  # <- custom border_width
                                                       fg_color=None,  # <- no fg_color
                                                       command=self.setting)
        self.button_settings.grid(row=0, column=2)

        # NUMBERS
        self.num_1 = customtkinter.CTkLabel(master=self.frame_up,
                                            text="I",
                                            text_color="green",
                                            text_font=("Roboto Bold", -100))  # font name and size in px
        self.num_1.grid(row=4, column=2, rowspan=3, columnspan=3, pady=10)

        self.num_2 = customtkinter.CTkLabel(master=self.frame_up,
                                            text="II",
                                            text_color="green",
                                            text_font=("Roboto Bold", -100))  # font name and size in px
        self.num_2.grid(row=4, column=8, rowspan=3, columnspan=3, pady=10)

        self.num_3 = customtkinter.CTkLabel(master=self.frame_up,
                                            text="III",
                                            text_color="green",
                                            text_font=("Roboto Bold", -100))  # font name and size in px
        self.num_3.grid(row=4, column=14, rowspan=3, columnspan=3, pady=10)

        # Three Text
        self.label1 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="Who are you\n" +
                                                  " looking for?",
                                             text_font=("Roboto Medium", -20),
                                             height=100,
                                             justify=tkinter.LEFT)
        self.label1.grid(column=2, row=7, rowspan=2, columnspan=3, sticky="nwe", pady=15)

        self.label2 = customtkinter.CTkLabel(master=self.frame_up,
                                             text=" Where are you\n" +
                                                  "searching them?",
                                             text_font=("Roboto Medium", -20),
                                             height=100,
                                             justify=tkinter.LEFT)
        self.label2.grid(column=8, row=7, rowspan=2, columnspan=3, sticky="nwe", pady=15)

        self.label3 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="     Where will the \n" +
                                                  "matched image be?",
                                             text_font=("Roboto Medium", -20),
                                             height=100,
                                             justify=tkinter.LEFT)
        self.label3.grid(column=14, row=7, rowspan=2, columnspan=3, sticky="nwe", pady=15)

        # Three Buttons
        self.button_left = customtkinter.CTkButton(master=self.frame_up,
                                                   text="GO",
                                                   text_font=("Roboto Bold", -16),
                                                   border_width=1,  # <- custom border_width
                                                   fg_color=None,  # <- no fg_color
                                                   command=self.button_step1)
        self.button_left.grid(row=9, column=2, columnspan=3, pady=20, padx=20, sticky="we")

        self.button_middle = customtkinter.CTkButton(master=self.frame_up,
                                                     text="GO",
                                                     text_font=("Roboto Bold", -16),
                                                     border_width=1,  # <- custom border_width
                                                     fg_color=None,  # <- no fg_color
                                                     command=self.button_step2)
        self.button_middle.grid(row=9, column=8, columnspan=3, pady=20, padx=20, sticky="we")

        self.button_right = customtkinter.CTkButton(master=self.frame_up,
                                                    text="GO",
                                                    text_font=("Roboto Bold", -16),
                                                    border_width=1,  # <- custom border_width
                                                    fg_color=None,  # <- no fg_color
                                                    command=self.button_step3)
        self.button_right.grid(row=9, column=14, columnspan=3, pady=20, padx=20, sticky="we")

    # Define all buttons
    def button_step1(self):
        try:
            global key_image_destination, key_image, key_face_encoding, known_face_encodings
            # Ask where the key image you wanna use is, and also their name
            key_image_destination = filedialog.askopenfilename(initialdir="/", title="Who are you looking for?",
                                                               filetypes=(("image", "*.jpg"), ("all file", "*.*")))

            # Load a sample picture and learn how to recognize it.
            key_image = face_recognition.load_image_file(key_image_destination)
            key_face_encoding = face_recognition.face_encodings(key_image)[0]

            # Create arrays of known face encoding and their name
            known_face_encodings = [
                key_face_encoding
            ]
        except AttributeError:
            print("Please select a file!")
        except IndexError:
            print("Bruh, there's no faces in the image")
        except PermissionError:
            print("Woah there, you don't have permission to open that")
        except UnidentifiedImageError:
            print("That's not an image you dummy")

    def button_step2(self):
        global unknown_images_location
        # Create arrays of unknown faces encoding
        unknown_images_location = glob(filedialog.askdirectory(title="Where are you searching them?") + "/*")

    def button_step3(self):
        global matched_image_destination
        # See where to put the matched one in
        matched_image_destination = filedialog.askdirectory(title="Where the matched images will be")

    def button_process(self):
        try:
            # Since the first image
            img_index = -1
            matched = 0
            print("Processing...")

            # Loop until the last image
            while img_index < len(unknown_images_location):
                global unknown_image
                try:
                    # Load an image with an unknown face
                    unknown_image = face_recognition.load_image_file(unknown_images_location[img_index])
                except (UnidentifiedImageError, PermissionError):  # Skip folders and unwanted files
                    pass
                # Find all the faces and face encodings in the unknown image
                face_locations = face_recognition.face_locations(unknown_image)
                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                # Loop through each face found in the unknown image
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        shutil.move(unknown_images_location[img_index], matched_image_destination)
                        matched = matched + 1

                # So we can move on
                img_index = img_index + 1

            # See how many image you get
            if matched == 0:
                print("I found no matches for you...")
            else:
                print("I found " + str(matched) + " face(s) that match(es) your selected face")
        except IndexError:
            print("The folder you're searching is empty")
        except NameError:
            print("No images found in the searching folder")
        except FileNotFoundError:
            print("Please complete all steps properly before performing a new search")

    def show_main(self):
        self.clearFrame()
        self.frame_main()

    def clearFrame(self):
        # Destroy all widgets from frame
        for widget in App.winfo_children(self):
            widget.destroy()

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def setting(self):
        self.clearFrame()
        # configure grid layout (2x1)
        self.grid_columnconfigure(0, weight=1)  # empty column  as spacing
        self.grid_rowconfigure(0, weight=1)  # empty row as spacing

        self.setting_frameright = customtkinter.CTkFrame(master=self)
        self.setting_frameright.grid(row=0, column=1, sticky="nswe")
        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.setting_frameright,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=20, padx=20, sticky="w")

        self.button_back = customtkinter.CTkButton(master=self.setting_frameright,
                                                   text="BACK",
                                                   text_font=("Roboto Light", -14),  # font name and size in px
                                                   command=self.show_main)
        self.button_back.grid(row=1, column=2, pady=10, padx=20)

    def on_closing(self):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
