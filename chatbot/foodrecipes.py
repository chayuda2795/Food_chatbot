import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

recipes = [
    {"title": "Spaghetti Carbonara", "ingredients": "spaghetti, eggs, parmesan cheese, pancetta, garlic, black pepper", "instructions": "Cook spaghetti. Mix eggs and cheese. Cook pancetta with garlic. Combine everything."},
    {"title": "Chicken Curry", "ingredients": "chicken, curry powder, coconut milk, onions, garlic, ginger", "instructions": "Cook chicken with onions, garlic, and ginger. Add curry powder and coconut milk. Simmer."},
    {"title": "Caesar Salad", "ingredients": "romaine lettuce, croutons, parmesan cheese, caesar dressing", "instructions": "Toss lettuce with dressing. Add croutons and cheese."},
    # Add more recipes as needed
]

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the recipes
recipe_texts = [f"{recipe['title']} {recipe['ingredients']}" for recipe in recipes]
recipe_embeddings = model.encode(recipe_texts)

# Create a FAISS index
dimension = recipe_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(recipe_embeddings)

def retrieve_recipe(query, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_recipes = [recipes[idx] for idx in indices[0]]
    return retrieved_recipes

def chatbot():
    print("Welcome to the Food Recipe Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        retrieved_recipes = retrieve_recipe(user_input)
        if retrieved_recipes:
            recipe = retrieved_recipes[0]
            print(f"Bot: Here's a recipe for {recipe['title']}:")
            print(f"Ingredients: {recipe['ingredients']}")
            print(f"Instructions: {recipe['instructions']}")
        else:
            print("Bot: Sorry, I couldn't find any recipes matching your query.")

# Run the chatbot
chatbot()