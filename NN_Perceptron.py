import pandas as pd
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import *
import tkinter.font as tkFont
import numpy as np

def CreateForm(size, title):
    form = Tk()
    form.geometry(size)
    form.title(title)
    form.resizable(False, False)
    return form


def CreateLabel(text):
    lbl_var = StringVar()
    lbl_var.set(text)
    lbl_features = Label(master=master, textvariable=lbl_var, font=fontStyle)
    lbl_features.pack(fill='x', padx=5, pady=5)


def CreateComboBox(data):
    cmb_obj = ttk.Combobox(master=master, textvariable=StringVar(), font=fontStyle)
    cmb_obj['values'] = data
    cmb_obj['state'] = 'readonly'
    cmb_obj.pack(fill='x', padx=5, pady=5)
    return cmb_obj


def CreateTextbox():
    txt_entry = Entry(master, font=fontStyle)
    txt_entry.pack(fill='x', padx=5, pady=5)
    return txt_entry


def CreateCheckbox(text):
    cb_var = IntVar()
    cb_obj = Checkbutton(master=master, text=text, variable=cb_var, onvalue=1, offvalue=0, font=fontStyle)
    cb_obj.pack()
    return cb_var


def GetSelectedFeatures():
    selected_option = features_cmb.get()
    selected_features = selected_option.split()
    selected_features.remove('and')
    return selected_features


def GetSelectedClasses():
    selected_option = classes_cmb.get()
    selected_classes = selected_option.split()
    selected_classes.remove('and')
    return selected_classes


def GetLearningRate():
    return float(txt_LearningRate.get())


def GetEpochNumber():
    return int(txt_Epochs.get())


# def GetThreshold():
#     return float(txt_Thresh.get())


def GetBiasDesicion():
    return int(use_bias.get())


def Draw_Iris_Dataset(class_a, class_b, class_c):
    x1_setosa = class_a["X1"]
    x2_setosa = class_a["X2"]
    x3_setosa = class_a["X3"]
    x4_setosa = class_a["X4"]

    x1_versicolor = class_b["X1"]
    x2_versicolor = class_b["X2"]
    x3_versicolor = class_b["X3"]
    x4_versicolor = class_b["X4"]

    x1_virginica = class_c["X1"]
    x2_virginica = class_c["X2"]
    x3_virginica = class_c["X3"]
    x4_virginica = class_c["X4"]

    plt.figure("figure1")
    plt.scatter(x1_setosa, x2_setosa)
    plt.scatter(x1_versicolor, x2_versicolor)
    plt.scatter(x1_virginica, x2_virginica)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    plt.figure("figure2")
    plt.scatter(x1_setosa, x3_setosa)
    plt.scatter(x1_versicolor, x3_versicolor)
    plt.scatter(x1_virginica, x3_virginica)
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.show()

    plt.figure("figure3")
    plt.scatter(x1_setosa, x4_setosa)
    plt.scatter(x1_versicolor, x4_versicolor)
    plt.scatter(x1_virginica, x4_virginica)
    plt.xlabel("X1")
    plt.ylabel("X4")
    plt.show()

    plt.figure("figure4")
    plt.scatter(x2_setosa, x3_setosa)
    plt.scatter(x2_versicolor, x3_versicolor)
    plt.scatter(x2_virginica, x3_virginica)
    plt.xlabel("X2")
    plt.ylabel("X3")
    plt.show()

    plt.figure("figure5")
    plt.scatter(x2_setosa, x4_setosa)
    plt.scatter(x2_versicolor, x4_versicolor)
    plt.scatter(x2_virginica, x4_virginica)
    plt.xlabel("X2")
    plt.ylabel("X4")
    plt.show()

    plt.figure("figure6")
    plt.scatter(x3_setosa, x4_setosa)
    plt.scatter(x3_versicolor, x4_versicolor)
    plt.scatter(x3_virginica, x4_virginica)
    plt.xlabel("X3")
    plt.ylabel("X4")
    plt.show()


def signum(val):
    if val >= 0:
        return 1
    else:
        return -1


def calculateNetValue(input, weight):
    return input.dot(weight.transpose())


def trainPerceptron(weight, data, epochs, lr):

    for epoch in range(epochs):
        correct = 0
        for i in range(len(data)):
            input = data.iloc[i, :3].to_numpy()
            target = data.iloc[i, -1]
            net = calculateNetValue(input, weight)
            y = signum(net)
            if y != target:
                e = target - y
                weight = weight + (lr*e*input)
            else:
                correct += 1
        if correct == len(data):
            break
    return weight


def drawLine(weight, class_a_1, class_a_2, class_b_1, class_b_2):
    # Equation: W1.X1+W2.X2+y = 0 ---> y = bias.
    # Here Calculate The other point coordinates
    point_1_y, point_2_x = 0, 10
    point_1_x = (-weight[2]*point_1_y - weight[0]) / weight[1]
    point_2_y = (-weight[1]*point_2_x - weight[0]) / weight[2]
    x_values = [point_1_x, point_2_x]
    y_values = [point_1_y, point_2_y]
    plt.scatter(class_a_1, class_a_2)
    plt.scatter(class_b_1, class_b_2)
    plt.plot(x_values, y_values)
    plt.show()

    return


def predict(input, weight):

    predictedValue = calculateNetValue(input, weight)
    out = signum(predictedValue)
    return out


def confusionMatrix(actual_output, prediction):

    # Defined Variables
    columns = ['actual', 'prediction', 'status']
    row = list()
    test_status = pd.DataFrame(columns=columns)
    confusion_matrix = np.zeros([2, 2])

    # Creating Confusion Matrix Process and Get status of output
    for i in range(len(prediction)):
        if actual_output[i] == 1:
            if prediction[i] == 1:
                confusion_matrix[0][0] += 1
                row.append([actual_output[i], prediction[i], 'Matching'])
            else:
                confusion_matrix[0][1] += 1
                row.append([actual_output[i], prediction[i], 'Mismatching'])

        elif actual_output[i] == -1:
            if prediction[i] == -1:
                confusion_matrix[1][1] += 1
                row.append([actual_output[i], prediction[i], 'Matching'])
            else:
                confusion_matrix[1][0] += 1
                row.append([actual_output[i], prediction[i], 'Mismatching'])

    # Then Print status of each output
    rows = pd.DataFrame(row, columns=columns)
    test_status = test_status.append(rows, ignore_index=True)
    print("--> The Test is Now Running ...")
    print(test_status)

    # Calculate accuracy by [sum of diagonal / total sum]
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # Show Confusion Matrix
    print("--> Confusion Matrix:")
    print(confusion_matrix)

    # return the accuracy
    return accuracy


def RunWholeProgram():

    dataset = pd.read_csv("IrisData.txt")
    output_header = dataset.columns[-1]
    class_a, class_b, class_c = dataset[:50], dataset[50:100], dataset[100:150]
    Draw_Iris_Dataset(class_a, class_b, class_c)

    # Get Results From Form
    user_features = GetSelectedFeatures()
    user_classes = GetSelectedClasses()
    user_lr = GetLearningRate()
    user_epochs = GetEpochNumber()
    user_bias = GetBiasDesicion()
    print(user_features[0])
    print(user_classes[1])
    print(user_lr)
    print(user_epochs)
    print(user_bias)

    # Remove un-needed flower Class
    flowers = {'C1': 'Iris-setosa', 'C2': 'Iris-versicolor', 'C3': 'Iris-virginica'}
    needed_flowers = list()
    for flower_class, flower_name in flowers.items():
        if flower_class not in user_classes:
            dataset = dataset[dataset[output_header] != flower_name]
        else:
            needed_flowers.append(flower_name)

    # Replace the flower name with 1 and -1 respectively
    numerical_output = [1, -1]
    dataset[output_header] = dataset[output_header].replace(needed_flowers, numerical_output)

     # Filter the dataset to selected features
    all_features = ['X1', 'X2', 'X3', 'X4']
    not_needed_features = list()
    needed_features = list()
    for feature in all_features:
        if feature not in user_features:
            not_needed_features.append(feature)
        else:
            needed_features.append(feature)
    dataset = dataset.drop(not_needed_features, axis=1)

    #Get Classes Values then Shuffle it
    class_a, class_b = dataset[:50], dataset[50:]

    # Then Split The data into 30 Train and 20 Test
    class_a_train, class_a_test = class_a[:30], class_a[30:]
    class_b_train, class_b_test = class_b[:30], class_b[30:]

    # Merge trained data together and tested data together
    trained_data = pd.concat([class_a_train, class_b_train])
    tested_data = pd.concat([class_a_test, class_b_test])

    trained_data = trained_data.sample(frac=1)
    tested_data = tested_data.sample(frac=1)

    if user_bias == 1:
        trained_data.insert(0, "X0", 1)
        tested_data.insert(0, "X0", 1)
    else:
        trained_data.insert(0, "X0", 0)
        tested_data.insert(0, "X0", 0)

    weight = np.random.rand(3)
    #train dataset
    actualWeight = trainPerceptron(weight, trained_data, user_epochs, user_lr)

    #print(actualWeight)

    #draw line to discrimnate between data
    drawLine(actualWeight, class_a[needed_features[0]], class_a[needed_features[1]], class_b[needed_features[0]], class_b[needed_features[1]])

    #get predicted values
    predection = list()
    actualOutput = list()
    for i in range(len(tested_data)):
        input = tested_data.iloc[i, :3].to_numpy()
        target = tested_data.iloc[i, -1]
        actualOutput.append(target)
        y_predected = predict(input, actualWeight)
        predection.append(y_predected)

    #build confusion matrix and calculate accuracy
    testAccuracy = confusionMatrix(actualOutput, predection)
    print("--> Accuracy: {}".format(testAccuracy))


master = CreateForm(size="350x450", title="TASK 2 SOLUTION")
fontStyle = tkFont.Font(family="JetBrains Mono", size=10)

# Create Label and Combobox for features
features = ('X1 and X2', 'X1 and X3', 'X1 and X4', 'X2 and X3', 'X2 and X4', 'X3 and X4')
CreateLabel(text="Select The Features")
features_cmb = CreateComboBox(data=features)

# Create Label and Combobox for Classes
classes = ('C1 and C2', 'C1 and C3', 'C2 and C3')
CreateLabel(text="Select The Classes")
classes_cmb = CreateComboBox(data=classes)

# Create Label and TextBox for Learning Rate
CreateLabel(text="Enter Learning rate Value")
txt_LearningRate = CreateTextbox()

# Create Label and TextBox for Epochs
CreateLabel(text="Enter Epochs Number")
txt_Epochs = CreateTextbox()

# Create Label and TextBox for Epochs
# CreateLabel(text="Enter MSE Threshold")
# txt_Thresh = CreateTextbox()

# Create Label and Checkbox for bias
CreateLabel(text="Please Check this for Selecting Bias")
use_bias = CreateCheckbox(text="Use Bias")

# Create Button
btn_submit = Button(master=master, text="Submit", width=15, command=RunWholeProgram, font=fontStyle)
btn_submit.pack(pady=15)

# Run the Form
master.mainloop()





