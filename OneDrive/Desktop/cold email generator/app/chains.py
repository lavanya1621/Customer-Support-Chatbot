import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ###SCARPED TEXT FROM WEBSITE:
            {page_data}
            ###INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: 'role', 'experience', 'skills' 
            Only return the valid JSON
            ###VALID JSON(NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res=chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res=json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context is too big, Unable to parse jobs")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self,job,links):
        prompt_email= PromptTemplate.from_template(
            """
            ###JOB DESCRIPTION:
            {job_description}

            ###INSTRUCTION:
            You are Lova, a business development executive at Beach and Coffee Co. It is an AI and software company seamless integration
            of business processes through automated technologies.
            Your job is to write a cold email to the client regarding the job mentioned above to fulfill their needs.
            also add most relevant ones from the  following links to showcase Beach and Cofee Co. portfolio: {link_list}
            Remember u are Lova,  A BDE AT BEACH AND COFFEE CO.
            dont need preamble

            ###EMAIL (NO PREAMBLE):
            """
        )

        chain_email= prompt_email | self.llm
        res = chain_email.invoke({"job_description" : str(job), "link_list": links})
        print(f"Response from LLM:{res}")
        #print(res.content)

        if res and hasattr(res, 'content'):
            email_content = res.content
            print(email_content)  # Debugging: print to the console
            return email_content  # Return the content of the email
        else:
            print("Error: No content in response.")  # Debugging: Handle missing content
            return "No email content generated."
    
if __name__ == "__main__":
    # Create instance of Chain
    chain = Chain()
    
    # Sample job and links for testing
    job = {
        "role": "Software Engineer",
        "experience": "5 years",
        "skills": ["Python", "JavaScript"]
    }
    links = ["https://portfolio.com/project1", "https://portfolio.com/project2"]
    
    # Call the write_mail method to generate email
    email_content = chain.write_mail(job, links)
    
    # Print the generated email content
    print(email_content)  # This should display the email on the console



     
    




