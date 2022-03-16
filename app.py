# Import module
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import os
import face_recognition_models
import dlib
from tkinter import messagebox
from tkinter import filedialog


# Importation des models pré-entrainés de dlib

# fonction de détetcion du visage avec la méthode HOG: applique (HOG + Linear SVM) pour la reconnaissance faciale
face_detector = dlib.get_frontal_face_detector()

# model de sélection des 68 points correspondant au visage
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

# model de sélection des 5 points correspondant au visage //optionnel
#predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
#pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

# model Resnet v1 pour encodage des visages
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


cap = cv2.VideoCapture(0)

# Dossier images
path = 'Images'
images = []
names = []
myList = os.listdir(path)

tolerance = 0.6  # Tolérance de la reconnaissance


# Fonctions de traitement intérmaidiaire:

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

# Fonctions principales


def coord_visage(img, echantillonnage=1, model="hog"):
    return face_detector(img, echantillonnage)


def distance_entre_visages(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

# recupere le chemin vers le fichier


def file_opener(chemin):
    path = filedialog.askopenfilename(filetypes=[("Image File", '.jpg')])
    chemin.config(text=path)


# Fonction de generation des encodages : 128 points
def liste_encodages(images):
    codelist = []
    for img in images:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        num_it = 1

        face_locations = face_detector(img, 1)

        pose_predictor = model_landmarks_entrainement

        # Les 68 points pour chaque image
        landmarks = [pose_predictor(img, face_location)
                     for face_location in face_locations]

        # tableau de tableaux: des 128 codes
        encodage = [np.array(face_encoder.compute_face_descriptor(
            img, raw_landmark_set, num_it)) for raw_landmark_set in landmarks]

        encode = encodage[0]
        codelist.append(encode)

    return codelist
# fin de generation des encodages des images d'entrainement


# Recuperation des images d'entrainement
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])


model_landmarks_entrainement = pose_predictor_68_point


# Liste des encodages des visages connus a partir de la liste des images
liste_code_connu = liste_encodages(images)

# fonction de capture


def snapshot(cap, entree, listecodes):
    ret, frame = cap.read()
    cv2.imwrite("Images/" + entree.get() + ".jpg", frame)

    messagebox.showinfo(
        title="Succés", message="Image ajoutée! ")
# fin fonction de capture


# fonction ; ajouter une image a partir d'un fichier
def save_entr(chemin, entree):

    im = Image.open(chemin)
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    cv2.imwrite("Images/" + entree + ".jpg", im)
    global images
    global names
    global myList

    images = []
    names = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        names.append(os.path.splitext(cl)[0])

    global liste_code_connu
    liste_code_connu = liste_encodages(images)

    messagebox.showinfo(
        title="Succés", message="Image ajoutée! Apprentissage terminé")
# fin ajout a partir du fichier


# Effectuer entrainement manuel
def entrainer():
    global images
    global names
    global myList

    images = []
    names = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        names.append(os.path.splitext(cl)[0])

    global liste_code_connu
    liste_code_connu = liste_encodages(images)
    messagebox.showinfo(
        title="Succés", message="Apprentissage terminé")
# fin


def changertol(entreeTol):
    global tolerance
    tolerance = float(entreeTol)

    messagebox.showinfo(
        title="Succés", message="Tolérance changée!")


# Create object
root = Tk()

# dimension interface
root.geometry("1300x640")

# Add image file
bg = PhotoImage(file="start.png")
mg = PhotoImage(file="fond.png")

w = "1300"
h = "640"

# frames
startframe = LabelFrame(root)
startframe.place(x=0, y=0, width=w, height=h)

mainframe = LabelFrame(root)
mainframe.place(x=0, y=0, width=w, height=h)


# entrainemnt frame: ajout nouvelles personnes
infoframe = LabelFrame(root)
infoframe.place(x=0, y=0, width=w, height=h)


###########################   Manage start frame   ############################
# Create Canvas
canvas1 = Canvas(startframe, width=1300,
                 height=640)

canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")

# Add Text
#canvas1.create_text(200, 250, text="Welcome")

# Create Buttons
buttoncmc = Button(startframe, text="Commencer",
                   command=lambda: mainframe.tkraise())


btninfo = Button(startframe, text="Apprentissage",
                 command=lambda: infoframe.tkraise())

# Display Buttons
button1_cmc = canvas1.create_window(830, 350,
                                    anchor="nw",
                                    window=buttoncmc)

button1_info = canvas1.create_window(830, 450,
                                     anchor="nw",
                                     window=btninfo)


##########################   Manage the main frame   #############################


# Create Canvas
canvas2 = Canvas(mainframe, width=1300,
                 height=640)

canvas2.pack(fill="both", expand=True)

# Display image
canvas2.create_image(0, 0, image=mg,
                     anchor="nw")

# create buttons
btnRetourStart = Button(mainframe, text="Retour",
                        command=lambda: startframe.tkraise())

# btncapture = Button(mainframe, text="Capturer",
# command=lambda: snapshot(cap))

# display buttons
button1_rtr = canvas2.create_window(100, 10,
                                    anchor="nw",
                                    window=btnRetourStart)

btnEntr = Button(mainframe, text="Apprentissage",
                 command=lambda: entrainer())

button1_rtr = canvas2.create_window(1020, 350,
                                    anchor="nw",
                                    window=btnEntr)
#btncpt = canvas2.create_window(100, 10, anchor="nw",  window=btncapture)


#######################   Manage the info frame    #########################

# Create Canvas
canvas3 = Canvas(infoframe, width=1300,
                 height=640)

canvas3.pack(fill="both", expand=True)

# Display image
canvas3.create_image(0, 0, image=mg,
                     anchor="nw")

btnRetourStart2 = Button(infoframe, text="Retour",
                         command=lambda: startframe.tkraise())

# display buttons
button1_rtr2 = canvas3.create_window(100, 40,
                                     anchor="nw",
                                     window=btnRetourStart2)


frame31 = LabelFrame(infoframe, text="Espace apprentissage",
                     width=1000, height=300)
frame31.place(x=100, y=90)

# generer liste des images affichable
pathxx = 'Images'
imagesxx = []
myListxx = os.listdir(pathxx)

for cl in myListxx:
    curImg = ImageTk.PhotoImage(Image.open(
        f'{pathxx}/{cl}').resize((200, 200)))
    imagesxx.append(curImg)


scrollbar = Scrollbar(frame31, orient=VERTICAL)
scrollbar.pack(side=RIGHT, fill=Y)
mainsection = Canvas(
    frame31, bg="pink", yscrollcommand=scrollbar.set, width=1000, height=300)

mainsection.pack(side=LEFT, anchor=CENTER, fill=BOTH, expand=True)
scrollbar.config(command=mainsection.yview)


# pour chaque image: creer un libeler, lui affecter l'image, l'afficher
k = 0

length = len(imagesxx)
for i in range(length):

    l = Label(frame31)
    l['image'] = imagesxx[i]
    x = 0
    y = 350

    if (i % 3) == 0:
        x = 460
        k = k+250

    if (i % 3) == 1:
        x = 790

    if (i % 3) == 2:
        x = 1120

    button1_cmc = mainsection.create_window(x, y+k,
                                            anchor="nw",
                                            window=l)

# Creates disabled scrollbar
mainsection.configure(scrollregion=mainsection.bbox(ALL))


btnapprentissage = Button(infoframe, text="Apprentissage",
                          command=lambda: entrainer())
button1_rtr2 = canvas3.create_window(520, 420,
                                     anchor="nw",
                                     window=btnapprentissage)

frame3 = LabelFrame(infoframe,
                    text="Ajouter une image d'apprentissage")

frame3.place(x=100, y=490, width=1000)

chemin = Label(frame3, text="Chemin vers l'image", width=80)
chemin.pack()

# Button label
btnfileexp = Button(frame3, text='Parcourir ... ',
                    command=lambda: file_opener(chemin))

btnfileexp.pack()

entreeajt = Entry(frame3)
entreeajt.pack()


btnajt = Button(frame3, text="Ajouter à l'apprentissage",
                command=lambda: save_entr(chemin.cget("text"), entreeajt.get()))

btnajt.pack()


# initiate start frame
startframe.tkraise()


# create necessities
L1 = Label(mainframe, bg="red")

imagewindw = canvas2.create_window(100, 40,
                                   anchor="nw",
                                   window=L1)

labelframe_widget = LabelFrame(mainframe,
                               text="Général", width=500, height=400)


if (model_landmarks_entrainement == pose_predictor_68_point):
    x = " Predicteur 68 points"
else:
    x = " Predicteur 5 points"

frame2 = LabelFrame(mainframe,
                    text="Ajouter la personne par capture :")


labelframe_widget.place(x=800, y=80)
frame2.place(x=800, y=300)

echantillonnage = 1  # echantillonage dans la détection
model_landmark_detection = pose_predictor_68_point  # model landmarks


label_widget = Label(labelframe_widget,
                     text="Méthode de détéction des visages: HOG + LINEAR SVM \n Model landmark  = " + x + "\n Tolérence: ")
label_widget.pack()
textEntry = StringVar()

entree2 = Entry(labelframe_widget, textvariable=textEntry)
textEntry.set(str(tolerance))
entree2.pack()


btntl = Button(mainframe, text="Changer tolérance",
               command=lambda: changertol(entree2.get()))

buttonTol = canvas2.create_window(1000, 200,
                                  anchor="nw",
                                  window=btntl)


label_f2 = Label(frame2,
                 text="Nom de la personne: ")
label_f2.pack()

entree1 = Entry(frame2)
entree1.pack()

btncapture = Button(frame2, text="Capturer",
                    command=lambda: snapshot(cap, entree1, liste_code_connu))

btncapture.pack()


# Debut traitement de l'image actuelle
while True:
    # pour chaque image du flux video
    img = cap.read()[1]

    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    # Detection des visages dans l'image
    model2 = "hog"
    facesCurFrame = [_trim_css_to_bounds(_rect_to_css(
        face), imgs.shape) for face in coord_visage(imgs, echantillonnage, model2)]

    num_2it = 1
    face_image = imgs
    face_locations = facesCurFrame

    face_locations = [_css_to_rect(face_location)
                      for face_location in face_locations]

    pose_predictor = model_landmark_detection

    raw_landmarks = [pose_predictor(face_image, face_location)
                     for face_location in face_locations]

    encodesCurFrame = [np.array(face_encoder.compute_face_descriptor(
        imgs, raw_landmark_set, num_2it)) for raw_landmark_set in raw_landmarks]

    # Reconnaissance
    for encodeface, faceloc in zip(encodesCurFrame, facesCurFrame):

        # comparer la liste des codes connu à l'encodage généré => liste true/false
        matches = list(distance_entre_visages(
            liste_code_connu, encodeface) <= tolerance)

        # generer un vecteur des dictances euclediennes entre les encodages
        if len(liste_code_connu) == 0:
            faceDis = np.empty((0))
        else:
            faceDis = np.linalg.norm(liste_code_connu - encodeface, axis=1)

        # prendre l'indice de la plus petite distance
        matchIndex = np.argmin(faceDis)

        # Vérifier si il respecte la tolérence
        if matches[matchIndex]:
            name = names[matchIndex].upper()

            nb = 1-faceDis[matchIndex]
            nb = round(nb*100, 2)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0.255, 0), 2)
            cv2.rectangle(img, (x1, y2), (x2, y2+35),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name + " " + str(nb)+"%", (x1+6, y2+24),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0.255, 0), 2)
            cv2.rectangle(img, (x1, y2), (x2, y2+35),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "NON RECONNU", (x1+6, y2+24),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    L1['image'] = img
    # Execute tkinter
    root.update()
