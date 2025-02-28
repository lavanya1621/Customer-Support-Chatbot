import pandas as pd
import chromadb
import uuid

class Portfolio:
    def __init__(self, file_path="C:\\Users\\saila\\OneDrive\\Desktop\\cold-email generator\\my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        # Add data to the collection only if it's not already loaded
        if self.collection.count() > 0:
            print("Portfolio already loaded into the collection.")
            return

        print(f"The file path is: {self.file_path}")  
        print(f"The data is: {self.data}")

        for _, row in self.data.iterrows():
            techstack = row.get("Techstack")
            links = row.get("Links")
            
            if pd.isna(techstack) or not isinstance(techstack, str) or techstack.strip() == "":
                print(f"Skipping invalid Techstack: {techstack}")
                continue

            if pd.isna(links) or not isinstance(links, str) or links.strip() == "":
                print(f"Skipping invalid Links: {links}")
                continue

            try:
                self.collection.add(
                    documents=[techstack],
                    metadatas={"links": links},
                    ids=[str(uuid.uuid4())]
                )
            except Exception as e:
                print(f"Error adding document: {e}")

        print(f"Collection count after loading: {self.collection.count()}")

    def query_links(self, skills):
        # Ensure skills is a non-empty list
        if not isinstance(skills, list):
            skills = [skills]

        if not skills or all(not isinstance(skill, str) or skill.strip() == "" for skill in skills):
            print("Error: Skills list is empty or contains invalid entries.")
            return []

        try:
            # Perform the query
            query_results = self.collection.query(query_texts=skills, n_results=2)

            # Handle no results 
            if query_results.get('metadatas'):
                return query_results['metadatas']
            else:
                print("No matching metadata found for skills.")
                return []  # Return an empty list if no metadata is found
        except Exception as e:
            print(f"Error querying links: {e}")
            return []
