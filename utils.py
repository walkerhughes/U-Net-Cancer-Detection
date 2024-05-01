import matplotlib.pyplot as plt 

def plot_loss_and_accuracy(train_losses, train_accuracy, val_losses, val_accuracy): 

    plt.rcParams["figure.figsize"] = (17, 8)

    plt.subplot(121)
    a, b = zip(*train_losses)
    plt.plot(a, b, "o", alpha = 0.45, label = "Training")
    a, b = zip(*val_losses)
    plt.plot(a, b, label = "Validation")
    plt.title("Losses Data")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(122) 
    a, b = zip(*train_accuracy)
    plt.plot(a, [b.item() for b in b], "o", alpha = 0.45, label = "Training")
    a, b = zip(*val_accuracy)
    plt.plot(a, [b.item() for b in b], label = "Validation")
    plt.title("Accuracy Data")
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.savefig("losses_vs_accuracy.jpg", dpi = 200)
    plt.show()


def plot_predictions(model, validation_data, validation_image): 
    # Code for testing prediction on an image
    plt.rcParams["figure.figsize"] = (11, 11)

    # define the needed image from the data set 
    x, y_truth = validation_data[validation_image]
    x, y_truth = x.cuda(), y_truth.cuda() 

    # push through the model
    y_hat = model(x.unsqueeze(0).float())  

    # subplots for the true image mask and the predicted one from our model 
    plt.subplot(131)
    plt.imshow(validation_data[validation_image][0][0], cmap = "gray") 
    plt.imshow(validation_data[validation_image][0][1], cmap = "gray")
    plt.imshow(-validation_data[validation_image][0][2], cmap = "gray") 
    plt.title("Original Image")

    plt.subplot(132)
    plt.imshow(validation_data[validation_image][1], cmap = "gray")
    plt.title("Truth")

    plt.subplot(133)
    plt.imshow(y_hat.cpu().argmax(1)[0], cmap = "gray")
    plt.title("Predicted Image Segmentation")

    plt.savefig(f"prediction_image_{validation_image}.jpg")
    plt.show()