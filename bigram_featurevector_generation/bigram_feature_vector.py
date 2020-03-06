import requests


def feature_vector(b):
    # Generate bigrams from the given string
    bigrams = [b[i:i+2] for i in range(len(b)-1)]

    # Obtaining word Count for the given sentence's bigrams
    word_count = {}

    for word in bigrams:
        if word in word_count:
            word_count[word] = word_count[word]+1
        else:
            word_count[word] = 1

    # Converting the given bigram characters to their hexadecimal values
    hexes = [[format(ord(char), "x") for char in list(x)] for x in word_count.keys()]
    hexes_1 = []
    for item in hexes:
        temp = item[0] + item[1]
        hexes_1.append(temp)

    # Converting the given bigram hexadecimal values to their decimal values
    decimals = []
    for item in hexes_1:
        decimals.append(int(item, 16))

    # Combining the given bigram decimal values and their word count to obtain feature vector
    values = word_count.values()
    final_dict = dict(zip(decimals, values))
    final_dict = dict(sorted(final_dict.items(), key=lambda x: x[0]))

    # Returning the final feature vector
    return final_dict

if __name__=="__main__":
    # Obtain details from URL using GET https command
    response = requests.get("https://pastebin.com/raw/V5XpP3s0")

    # Obtain the data from response received - Note: Response will be received only when it is 200 else it will be 400
    sentence = response.text

    # Calling feature_vector function to convert the given string to its decimal values and obtain word count
    if sentence is not None:
        result = feature_vector("")
    else:
        print("Input is not valid string")
        exit(0)

    # Display the final results
    print(result.items())
