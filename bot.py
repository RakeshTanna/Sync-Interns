import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample training data
training_data = [
    "Hi, how are you?",
    r"I'm good. How about you?",
    "What is your name?",
    r"My name is Chatbot.",
    "Tell me a joke.",
    r"Sure! Why don't scientists trust atoms? Because they make up everything!",
    "What is the weather like today?",
    r"I'm sorry, I don't have access to real-time information.",
    "What are your hobbies?",
    r"I enjoy learning new things and helping people!",
    "How old are you?",
    r"As a chatbot, I don't have an age. I'm here to assist you.",
    # Add more training data examples here
]

# Initialize the tokenizer and stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = nltk.tokenize.WordPunctTokenizer()

# Preprocess the training data
preprocessed_data = []
for i in range(0, len(training_data), 2):
    question = training_data[i]
    response = training_data[i + 1]
    
    question_tokens = tokenizer.tokenize(question.lower())
    question_tokens = [token for token in question_tokens if token.isalpha() and token not in stop_words]
    preprocessed_data.append(' '.join(question_tokens))

    response_tokens = tokenizer.tokenize(response.lower())
    response_tokens = [token for token in response_tokens if token.isalpha() and token not in stop_words]
    preprocessed_data.append(' '.join(response_tokens))

# Initialize the vectorizer and transform the training data
vectorizer = TfidfVectorizer()
vectorized_data = vectorizer.fit_transform(preprocessed_data)

# Function to preprocess user input
def preprocess_input(user_input):
    tokens = tokenizer.tokenize(user_input.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    preprocessed_input = ' '.join(tokens)
    return preprocessed_input

# Function to generate a response
def generate_response(user_input):
    preprocessed_input = preprocess_input(user_input)
    input_vector = vectorizer.transform([preprocessed_input])
    similarity_scores = cosine_similarity(input_vector, vectorized_data)
    max_similarity_index = similarity_scores.argmax()
    response_index = max_similarity_index // 2
    response = training_data[response_index * 2 + 1]
    return response

# Main loop to get user input and generate responses
# Main loop to get user input and generate responses
while True:
    user_input = input('You: ')
    if user_input.lower() == 'bye':
        print('Bot: Goodbye!')
        break
    response = generate_response(user_input)
    print('Bot:', response)
