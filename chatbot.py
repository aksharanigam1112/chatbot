# from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import warnings
warnings.filterwarnings(action="ignore")

# app = Flask(__name__)

bot = ChatBot("Chatterbot",storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(bot)
trainer.train("chatterbot.corpus.english")

# @app.route("/")
# def home():
#     return render_template("index.html")
#
# @app.route("/get")
def get_bot_response(usertext):
    # usertext = input("Enter the message :-")
    return str(bot.get_response(usertext))

if __name__ == '__main__':
    # app.run()
    print("\n\nHi !! I'm Chatty, you can talk to me whenever you want !! \n")
    usertext = input("Enter the message :-")
    while(usertext != 'exit' and usertext !='bye' and usertext !='quit'):
        response = get_bot_response(usertext)
        print(response)
        usertext = input("Enter the message :- ")

