from typing import Literal, Dict, List
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
import random
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAwjHr_8ucLsNTutTvTm64JRnJCj5b_ngs"
llm = init_chat_model(model="google_genai:gemini-2.0-flash", temperature=0.7)

# Players
players = ["Alice", "Bob", "Charlie"]
topic_pool = ["Pizza", "Soccer", "Dinosaurs", "Space Travel", "Video Games"]
secret_topic = random.choice(topic_pool)
impostor = random.choice(players)

print(f"Secret topic: {secret_topic}, Impostor: {impostor}")

class GameState(BaseModel):
    step: int = 0
    secret: str = ""
    player_descriptors: Dict[str, List[str]] = {}  # map of player name to their descriptors
    descriptor_history: str = ""  # running conversation log
    current_turn: Literal["Alice", "Bob", "Charlie"] = "Alice"

# Abilities of players + storage
def node_fn(state: GameState, config: RunnableConfig) -> GameState:
    player_name = state.current_turn
    is_impostor = player_name == impostor
    player_descriptors = state.player_descriptors
    descriptor_history = state.descriptor_history

    prompt = f"""
You are playing a game where the topic is hidden from one player (the impostor).
The known topic is: {secret_topic if not is_impostor else '???'}.

You are {player_name}. Your goal is to give a short, vague but relevant descriptor that fits the topic (if you know it) or guess something plausible based on what others said.

Conversation so far:
{descriptor_history}

Your turn. Give one short descriptor (1 sentence or phrase) that could match the topic or help you blend in.
"""

    response = llm.invoke(prompt).content.strip()

    updated_descriptors = descriptor_history + f"\n{player_name}: {response}"

    updated_player_descriptors = dict(player_descriptors)
    if player_name not in updated_player_descriptors:
        updated_player_descriptors[player_name] = []
    updated_player_descriptors[player_name].append(response)

    next_player = players[(state.step + 1) % len(players)]

    return state.model_copy(
        update={
            "step": state.step + 1,
            "descriptor_history": updated_descriptors,
            "player_descriptors": updated_player_descriptors,
            "current_turn": next_player
        }
    )


def get_next_node(state: GameState) -> str:
    step = state.step
    if step >= 6:  # 2 rounds (3 players x 2)
        return "end"
    next_player = players[step % len(players)]
    return next_player

# Langgraph
graph = StateGraph(GameState)

for player in players:
    graph.add_node(player, node_fn)

graph.set_entry_point("Alice")

for player in players:
    graph.add_conditional_edges(player, get_next_node)

graph.add_node("end", lambda x: x)
graph.add_edge("end", END)

runnable = graph.compile()

initial_state = GameState(
    step=0,
    secret=secret_topic,
    current_turn="Alice",
    player_descriptors={},
    descriptor_history=""
)

# Run simulation
final_output = runnable.invoke(initial_state)

final_state = GameState(**final_output)

#output
print("\n--- Game Over ---")
print(f"Secret topic: {secret_topic}")
print(f"Impostor: {impostor}")
print("\nDescriptor History:")
print(final_state.descriptor_history)
print("\nEach player's descriptors:")
for player, descriptors in final_state.player_descriptors.items():
    print(f"{player}: {descriptors}")
