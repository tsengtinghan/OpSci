import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = "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"

def refine_description(description):
    prompt = f"Given the project: '{description}', summarize the skills required, levels of expertise needed, fields of research, and other important aspects. Please be concise, no more than two sentences."
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ])
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def main():
    # Load the CSV file
    df = pd.read_csv('feed.csv')

    # Generate refined descriptions
    df['refined_description'] = df['project_description'].apply(refine_description)

    # Save the updated dataframe
    df.to_csv('feed.csv', index=False)
    print("Refined descriptions generated and saved to feed.csv!")

if __name__ == "__main__":
    main()
