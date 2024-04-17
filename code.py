#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


dataset = tf.keras.preprocessing.image_dataset_from_directory("casting/")


# In[3]:


dataset.class_names


# In[38]:


class_name=dataset.class_names


# In[5]:


len(dataset)


# In[6]:


train_ds=dataset.take(35)


# In[7]:


test_ds=dataset.skip(35)


# In[8]:


len(test_ds)


# In[ ]:





# # Model

# In[48]:


model=tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(256,256,3)),
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(2,activation="sigmoid"),
])


# In[49]:


model.summary()


# In[50]:


model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)


# In[52]:


history=model.fit(
    train_ds,
    batch_size=32,
    epochs=5,
)


# # Testing

# In[53]:


model.evaluate(test_ds)

After running 10 epochs we get a 91% accuracy on the testing dataset which is good
# In[54]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)

    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[56]:


plt.figure(figsize=(20, 10))
for images, labels in test_ds:
    for i in range(6):
        ax = plt.subplot(2,3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i])
        actual_class = class_name[int(labels[i])]

        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")
    break


# In[60]:


act_lab=[]
pred_lab=[]
c=0
for images, labels in test_ds:
    for i in range(32):
        predicted_class, confidence = predict(model, images[i])
        actual_class = class_name[int(labels[i])]
        act_lab.append(actual_class)
        pred_lab.append(predicted_class)
        c=c+1
        if c==179:
            break
        print(c)


# In[62]:


from sklearn.metrics import confusion_matrix

con_mat=confusion_matrix(y_true=act_lab,y_pred=pred_lab)


# In[ ]:





# In[63]:


import seaborn as sns
sns.heatmap(con_mat, annot=True)


# In[65]:


from sklearn.metrics import classification_report
print('Classification Report:\n', classification_report(act_lab, pred_lab))

# Here we can see that everything like precision ,  recall f1 score is good we can trust on this model
# In[ ]:




