from flask import Blueprint, request
import torch

from fy_bot.gpt2_model import chat

message_blueprint = Blueprint(name="message_blueprint", import_name=__name__)


PROJECT_NAME = "tax_examples"

@message_blueprint.route("/", methods=["POST"])
def create_dataset():
    data = request.get_json()

    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    response = chat(PROJECT_NAME, data["message"], device)

    return { "response": response }
