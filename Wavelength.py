from typing import Literal, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
import random
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAwjHr_8ucLsNTutTvTm64JRnJCj5b_ngs"
llm = init_chat_model(model="google_genai:gemini-2.0-flash", temperature=0.7)

#Players
players = ["Alice", "Bob", "Charlie"]
topic_pool = ["Pizza", "Soccer", "Dinosaurs", "Space Travel", "Video Games"]
secret_topic = random.choice(topic_pool)
impostor = random.choice(players)

print(f"Secret topic: {secret_topic}, Impostor: {impostor}")

class GameState(BaseModel):
    step: int = 0
    secret: str = ""
    player_descriptors: Dict[str, List[str]] = {}
    descriptor_history: str = ""
    current_turn: Literal["Alice", "Bob", "Charlie"] = "Alice"
    impostor_guesses: List[str] = []
    player_reflections: Dict[str, List[str]] = {}
    votes: Dict[str, str] = {}  # who each player votes for

# Prompt 

def generate_prompt(state: GameState, player_name: str, is_impostor: bool) -> str:
    history = state.descriptor_history
    last_descriptor = history.splitlines()[-1] if history else ""

    if is_impostor:
        return f"""
You are {player_name}, the impostor.
You do not know the secret topic. Based on the previous descriptors:
{history}
Guess the secret topic and give a vague but relevant descriptor (maximum of 3 words) to blend in.
Then explain your guess in 10-15 words
"""
    else:
        reasoning = ""
        if state.step > 0:
            reasoning = "Also, reflect on the previous descriptors in 15-20 words. Mention if any seem suspicious or incorrect, and say who you suspect as the impostor."

        return f"""
You are {player_name}. You know the topic: {state.secret}.
Give a short descriptor (maximum of 3 words) that fits the topic.
Then explain your reasoning in 10-15 words: why you chose this descriptor.
{reasoning}
Descriptor history:
{history}
"""

# Abilities of players

def node_fn(state: GameState, config: RunnableConfig) -> GameState:
    player_name = state.current_turn
    is_impostor = player_name == impostor

    prompt = generate_prompt(state, player_name, is_impostor)
    response = llm.invoke(prompt).content.strip()

    updated_history = state.descriptor_history + f"\n{player_name}: {response}"
    updated_descriptors = dict(state.player_descriptors)
    updated_reflections = dict(state.player_reflections)

    if player_name not in updated_descriptors:
        updated_descriptors[player_name] = []
    updated_descriptors[player_name].append(response)

    if player_name not in updated_reflections:
        updated_reflections[player_name] = []
    updated_reflections[player_name].append(response)

    updated_guesses = list(state.impostor_guesses)
    if is_impostor:
        guess_lines = [line for line in response.splitlines() if "topic" in line.lower() or "guess" in line.lower()]
        if guess_lines:
            latest_guess = guess_lines[-1]
            guess = latest_guess.split()[-1].strip(".?!")
            if not updated_guesses or updated_guesses[-1].lower() != guess.lower():
                updated_guesses.append(guess)

    next_player = players[(state.step + 1) % len(players)]

    return state.model_copy(
        update={
            "step": state.step + 1,
            "descriptor_history": updated_history,
            "player_descriptors": updated_descriptors,
            "player_reflections": updated_reflections,
            "impostor_guesses": updated_guesses,
            "current_turn": next_player
        }
    )

def get_next_node(state: GameState) -> str:
    if state.step >= 6:
        return "vote"
    return players[state.step % len(players)]

# Voting

def vote_node(state: GameState) -> GameState:
    vote_results = {}
    for player in players:
        prompt = f"""
You are {player}. The conversation so far:
{state.descriptor_history}

Now vote for who you think is the impostor and briefly justify your vote.
Just output a single name and a 10-15 word justification.
"""
        vote = llm.invoke(prompt).content.strip().split()[0]
        vote_results[player] = vote
    return state.model_copy(update={"votes": vote_results})

# End result
def end_node(state: GameState):
    from collections import Counter

    vote_counter = Counter(state.votes.values())
    max_votes = max(vote_counter.values())
    candidates = [name for name, count in vote_counter.items() if count == max_votes]

    # Majority Vote
    if len(candidates) > 1:
        result = "Impostor wins (tie in votes)."
    elif candidates[0] == impostor:
        result = "Players win! Impostor was caught."
    else:
        result = "Impostor wins! They were not identified."

    print("\n--- Game Over ---")
    print(f"Secret topic: {secret_topic}")
    print(f"Impostor: {impostor}")
    print("\nDescriptor History:")
    print(state.descriptor_history)
    print("\nPlayer Reflections:")
    for player, comments in state.player_reflections.items():
        print(f"{player}: {comments}")
    print("\nVotes:")
    for player, vote in state.votes.items():
        print(f"{player} voted for: {vote}")
    print(f"\nOutcome: {result}")
    print(f"Impostor guess changes: {len(state.impostor_guesses)}")
    return state


graph = StateGraph(GameState)

for player in players:
    graph.add_node(player, node_fn)

graph.set_entry_point("Alice")

for player in players:
    graph.add_conditional_edges(player, get_next_node)

graph.add_node("vote", vote_node)
graph.add_conditional_edges("vote", lambda _: "end")

graph.add_node("end", end_node)
graph.add_edge("end", END)

runnable = graph.compile()

# Run the simulation
initial_state = GameState(
    step=0,
    secret=secret_topic,
    current_turn="Alice",
    player_descriptors={},
    descriptor_history="",
    impostor_guesses=[],
    player_reflections={},
    votes={}
)

runnable.invoke(initial_state)
