from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
import os
from .forms import UploadForm
from .models import Image
from PIL import Image as im
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

import tensorflow as tf
global model

# Create your views here.


def home(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            i = Image.objects.latest('id')
            print(i.img.url)
            return render(request, 'upload.html', {'form': form, 'imgurl': i.img.url, 'isup': '1'})

    else:
        form = UploadForm()
        return render(request, 'upload.html', {'form': form})


def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


model = tf.keras.models.load_model(
    'model.h5',
    custom_objects={'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy,
                    "GlorotUniform": tf.keras.initializers.glorot_uniform},

    compile=True
)


def predict(request):

    if request.POST:
        imageobj = Image.objects.latest('id')

        img_path = settings.BASE_DIR+imageobj.img.url
        img_path.replace('\\', '/')
        preprocessed_image = prepare_image(img_path)
        a = model.predict(preprocessed_image)
        a = a[0]

        a = [round(x, 6) for x in a]
        print(str(a))
        index = np.argsort(a)[-2]

        b = ['Actinic keratosis', 'Basal cell carcinoma', ' Benign keratosis',
             'Dermatofibroma', 'Melanoma', 'Melanocytic nevus', 'Vascular Lesions']
        n = b[index]
        info = ['These small, scaly patches are caused by too much sun, and commonly occur on the head, neck, or hands, but can be found elsewhere. They can be an early warning sign of skin cancer, but it’s hard to tell whether a particular patch will continue to change over time and become cancerous. Most do not, but doctors recommend early treatment to prevent the development of squamous cell skin cancer.  Fair-skinned, blond, or red-haired people with blue or green eyes are most at risk.',
                'Basal cells produce new skin cells as old ones die. Limiting sun exposure can help prevent these cells from becoming cancerous.This cancer typically appears as a white, waxy lump or a brown, scaly patch on sun-exposed areas, such as the face and neck.Treatments include prescription creams or surgery to remove the cancer. In some cases radiation therapy may be required.',
                'A seborrhoeic keratosis is one of the most common non-cancerous skin growths in older adults. While it\'s possible for one to appear on its own, multiple growths are more common.Seborrheic keratosis often appears on the face, chest, shoulders or back. It has a waxy, scaly, slightly elevated appearance.No treatment is necessary. If the seborrhoeic keratosis causes irritation, it can be removed by a doctor.',
                'Dermatofibroma (superficial benign fibrous histiocytoma) is a common cutaneous nodule of unknown etiology that occurs more often in women. Dermatofibroma frequently develops on the extremities (mostly the lower legs) and is usually asymptomatic, although pruritus and tenderness can be present.',
                'Melanoma occurs when the pigment-producing cells that give colour to the skin become cancerous.Symptoms might include a new, unusual growth or a change in an existing mole. Melanomas can occur anywhere on the body.Treatment may involve surgery, radiation, medication or in some cases, chemotherapy.',
                'Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants, which all are included in our series. The variants may differ significantly from a dermatoscopic point of view.',
                'Cutaneous vascular lesions comprise of all skin disease that originate from or affect blood or lymphatic vessels, including malignant or benign tumors, malformations and inflammatory disease. While some vascular lesions are easily diagnosed clinically and dermoscopically, other vascular lesions can be challenging as many of them share similar dermoscopic features.',
                ]
        symtoms = ['scaly patch on skin or itching', 'lesion, redness, loss of colour, small bump, swollen blood vessels in the skin, or ulcers', ' itching, small bump on skin, or waxy elevated skin lesion', 'Usually appear on the lower legs, but may appear on the arms or trunk.May be red, pink, purplish, gray or brown and may change color over time.',
                   'bigger mole diameter, darkening of the skin, mole color changes, or skin mole with irregular border', 'birthmark or discolouration.Melanocytic nevus can be rough, flat or raised.', 'appear at birth as flat pink areas of skin discoloration and usually darken over time to become a deep purple color.']
        i = info[index]
        s = symtoms[index]
        index = np.argmax(a)

        return render(request, 'upload.html', {'isup': '1', 'imgurl': imageobj.img.url, 'n': n, 'i': i, 's': s})
    else:
        return HttpResponse("Error")
