import tkinter as tk
import tkinter.messagebox
import customtkinter
import face_recognition
import numpy as np
from tkinter import filedialog
from glob import glob
import shutil
from PIL import Image, ImageTk, UnidentifiedImageError

# In case you encountered import error | pip3 install dlib --force-reinstall --no-cache-dir --global-option=build_ext

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    width = 650
    height = 480

    def __init__(self):
        super().__init__()
        self.title("Face Finder")

        # Center the window
        screen_width = App.winfo_screenwidth(self)
        screen_height = App.winfo_screenheight(self)
        x = (screen_width / 2) - (App.width / 2)
        y = (screen_height / 2) - (App.height / 2)
        self.geometry(f"{App.width}x{App.height}+{int(x)}+{int(y)}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        self.iconphoto(True, tk.PhotoImage(file='assets/facedetectdark.png'))  # App icon
        self.frame_main()
        self.attributes('-alpha', 1.0)  # For transparency adjustment
        self.resizable(height=0, width=0)

    def frame_main(self):

        # ============ create down frame ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=0)  # reset from the setting part
        self.grid_columnconfigure(0, weight=1)  # empty column as spacing
        self.grid_rowconfigure(0, weight=1)  # empty row as spacing

        self.frame_down = customtkinter.CTkFrame(master=self,
                                                 height=120,
                                                 corner_radius=0)
        self.frame_down.grid(row=1, column=0, sticky="nswe")

        # ============ down frame components ============

        # configure grid layout (3x1)
        self.frame_down.grid_columnconfigure(1, weight=1)  # empty row as spacing
        self.frame_down.grid_rowconfigure((0, 2), minsize=5)

        # Things happen in the downside
        self.label_left = customtkinter.CTkLabel(master=self.frame_down,
                                                 text="Face Finder",
                                                 text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_left.grid(row=1, column=0, pady=10)

        self.button_process = customtkinter.CTkButton(master=self.frame_down,
                                                      text="Process",
                                                      text_font=("Roboto Light", -14),  # font name and size in px
                                                      command=self.button_process)
        self.button_process.grid(row=1, column=2, pady=10, padx=20)

        # ============ create up frame ============

        self.frame_up = customtkinter.CTkFrame(master=self)
        self.frame_up.grid(row=0, column=0, sticky="nswe", padx=20, pady=20)

        # ============ up frame components ============

        self.frame_up.columnconfigure((1, 6, 7, 12, 13, 17), weight=1)
        self.frame_up.grid_rowconfigure(8, weight=1)

        # Settings button
        self.button_settings = customtkinter.CTkButton(master=self.frame_up,
                                                       text="≡",
                                                       text_font=("Roboto Medium", -50),
                                                       width=75,
                                                       fg_color=None,  # <- no fg_color
                                                       command=self.settings)
        self.button_settings.grid(row=0, column=0, columnspan=3, sticky="w")  # I take the whole 19 columns

        # ICON
        self.icon_1 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="👤",
                                             text_font=("Roboto Bold", -100))  # font name and size in px
        self.icon_1.grid(row=1, column=2, rowspan=3, columnspan=3, pady=10)

        self.icon_2 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="👥",
                                             text_font=("Roboto Bold", -100))  # font name and size in px
        self.icon_2.grid(row=1, column=8, rowspan=3, columnspan=3, pady=10)

        self.icon_3 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="📁",
                                             text_font=("Roboto Bold", -100))  # font name and size in px
        self.icon_3.grid(row=1, column=14, rowspan=3, columnspan=3, pady=10)

        # Three Text
        self.label1 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="Who are you\n" +
                                                  " looking for?",
                                             text_font=("Roboto Medium", -20),
                                             height=100,
                                             justify=tkinter.LEFT)
        self.label1.grid(column=2, row=7, rowspan=3, columnspan=3, sticky="nwe", pady=15)

        self.label2 = customtkinter.CTkLabel(master=self.frame_up,
                                             text=" Where are you\n" +
                                                  "searching them?",
                                             text_font=("Roboto Medium", -20),
                                             height=100,
                                             justify=tkinter.LEFT)
        self.label2.grid(column=8, row=7, rowspan=3, columnspan=3, sticky="nwe", pady=15)

        self.label3 = customtkinter.CTkLabel(master=self.frame_up,
                                             text="     Where will the \n" +
                                                  "matched image be?",
                                             text_font=("Roboto Medium", -20),
                                             height=100,
                                             justify=tkinter.LEFT)
        self.label3.grid(column=14, row=7, rowspan=3, columnspan=3, sticky="nwe", pady=15)

        # Three Buttons
        self.button_left = customtkinter.CTkButton(master=self.frame_up,
                                                   text="GO",
                                                   text_font=("Roboto Bold", -16),
                                                   width=100, height=40,
                                                   border_width=1,  # <- custom border_width
                                                   fg_color=None,  # <- no fg_color
                                                   command=self.button_step1)
        self.button_left.grid(row=9, column=2, columnspan=3, pady=20, padx=20)

        self.button_middle = customtkinter.CTkButton(master=self.frame_up,
                                                     text="GO",
                                                     width=100, height=40,
                                                     text_font=("Roboto Bold", -16),
                                                     border_width=1,  # <- custom border_width
                                                     fg_color=None,  # <- no fg_color
                                                     command=self.button_step2)
        self.button_middle.grid(row=9, column=8, columnspan=3, pady=20, padx=20)

        self.button_right = customtkinter.CTkButton(master=self.frame_up,
                                                    text="GO",
                                                    width=100, height=40,
                                                    text_font=("Roboto Bold", -16),
                                                    border_width=1,  # <- custom border_width
                                                    fg_color=None,  # <- no fg_color
                                                    command=self.button_step3)
        self.button_right.grid(row=9, column=14, columnspan=3, pady=20, padx=20)

    def showwarning(self, msg):
        tkinter.messagebox.showwarning(title=None, message=msg)

    def showinfo(self, msg):
        tkinter.messagebox.showinfo(title=None, message=msg)

    # Define all buttons
    def button_step1(self):
        global key_image_destination, key_image, key_face_encoding, known_face_encodings
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
            self.showwarning("Please select a file!")
        except IndexError:
            self.showwarning("Bruh, there's no faces in the image")
        except PermissionError:
            self.showwarning("Woah there, you don't have permission to open that")
        except UnidentifiedImageError:
            self.showwarning("That's not an image you dummy")

    def button_step2(self):
        global unknown_images_location
        # Create arrays of unknown faces encoding
        unknown_images_location = glob(filedialog.askdirectory(title="Where are you searching them?") + "/*")

    def button_step3(self):
        global matched_image_destination
        # See where to put the matched one in
        matched_image_destination = filedialog.askdirectory(title="Where the matched images will be")

    def result(self):
        if matched == 0:
            self.showinfo("I found no matches for you, try complete each step properly")
        else:
            self.showinfo("I found " + str(matched) + " face(s) that match(es) your selected face")

    def button_process(self):
        global matched
        try:
            # Since the first image
            img_index = -1
            matched = 0
            print("Processing...")

            # Loop until the last image
            while enumerate(unknown_images_location):
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
                        try:
                            shutil.move(unknown_images_location[img_index], matched_image_destination)
                            matched += 1
                        except shutil.Error:  # Avoid same file exists error
                            pass
                # So we can move on
                img_index += 1

            # See how many image you get
            self.result()
        except IndexError:  # Some faces might not be detected
            self.result()
        except (TypeError, NameError):
            self.showwarning("Please complete all steps properly first")
        except FileNotFoundError:  # Avoid redundant process
            self.result()

    def show_main(self):
        self.clearFrame()
        self.frame_main()

    def clearFrame(self):
        # Destroy all widgets from frame
        for widget in App.winfo_children(self):
            widget.destroy()

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
        if new_appearance_mode == "Light":
            self.iconphoto(True, tk.PhotoImage(file='assets/facedetectlight.png'))
        else:
            self.iconphoto(True, tk.PhotoImage(file='assets/facedetectdark.png'))

    def change_color_theme(self, color_theme):
        customtkinter.set_default_color_theme(color_theme)

    def settings(self):
        self.clearFrame()

        # Frame up
        self.setting_frameup = customtkinter.CTkFrame(master=self)
        self.setting_frameup.grid(row=0, column=0, sticky="nswe", padx=20, pady=20)

        self.label_appearance = customtkinter.CTkLabel(master=self.setting_frameup,
                                                       text="Appearance Mode:",
                                                       text_font=("Roboto Bold", -14))
        self.label_appearance.grid(row=0, column=0, pady=10, padx=20, sticky="w")

        self.option_appearance = customtkinter.CTkOptionMenu(master=self.setting_frameup,
                                                             values=["System", "Light", "Dark"],
                                                             command=self.change_appearance_mode)
        self.option_appearance.grid(row=1, column=0, padx=20, sticky="w")

        self.label_appearance = customtkinter.CTkLabel(master=self.setting_frameup,
                                                       text="Color Theme:",
                                                       text_font=("Roboto Bold", -14))
        self.label_appearance.grid(row=0, column=1, pady=10, padx=20, sticky="w")

        self.option_appearance = customtkinter.CTkOptionMenu(master=self.setting_frameup,
                                                             values=["dark-blue", "blue", "green"],
                                                             command=self.change_color_theme)
        self.option_appearance.grid(row=1, column=1, padx=20, sticky="w")

        self.label_transparency = customtkinter.CTkLabel(master=self.setting_frameup,
                                                         text="Transparency",
                                                         text_font=("Roboto Bold", -14))
        self.label_transparency.grid(row=0, column=2, pady=10, padx=20, sticky="w")

        self.transparency = customtkinter.CTkSlider(master=self.setting_frameup,
                                                    from_=0.3,
                                                    to=1,
                                                    command=self.slide)
        self.transparency.grid(row=1, column=2, columnspan=2, pady=10, padx=10, sticky="we")
        self.transparency.set(self.attributes('-alpha'))  # Track the transparency

        # Description

        # Create a photoimage object of the image in the path
        credit = ImageTk.PhotoImage(Image.open("assets/credit.png").convert("RGBA"))
        labelcredit = tkinter.Label(master=self.setting_frameup, image=credit)
        labelcredit.image = credit
        labelcredit.grid(row=2, column=0, columnspan=5, pady=40, padx=3, sticky="we")  # Position image

        # Frame Down
        self.setting_framedown = customtkinter.CTkFrame(master=self,
                                                        height=120,
                                                        corner_radius=0)
        self.setting_framedown.grid(row=1, column=0, sticky="nswe")

        self.button_back = customtkinter.CTkButton(master=self.setting_framedown,
                                                   text="BACK",
                                                   text_font=("Roboto Light", -14),  # font name and size in px
                                                   command=self.show_main)
        self.button_back.grid(row=0, column=0, pady=15, padx=20)

    def slide(self, x):
        self.attributes('-alpha', self.transparency.get())

    def on_closing(self):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
