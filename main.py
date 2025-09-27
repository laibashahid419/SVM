from src.inference import predict_text


if __name__  == "__main__":
    print("Sentiment Analysis")

    user_input = input("Enter text for sentiment Analysis ")
    prediction = predict_text(user_input)


    if prediction ==0:
        print(" prediction Negtive")

    else :
           print(" prediction Postive")

