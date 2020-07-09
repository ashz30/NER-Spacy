import spacy

model_dir = "C:/Sharepoint/OneDrive - Blue Prism/impDocuments/hackathon/unstructureddata/model"
Lines = []

#custom model extraction
nlp = spacy.load(model_dir)

#uncomment for prebuilt model extraction
#nlp = spacy.load("en_core_web_sm")

def extract(data):
    count = 0
    extracted_data = []
    Lines = str(data).splitlines()
    for line in Lines:
        # Strip the newline character
        text = line.strip()
        doc = nlp(text)
        count= count + 1
        for entity in doc.ents:
            #print("with "+ model + " model at Line " + str(count) + " :" + entity.label_, ' | ', entity.text)
            #format to access - 0 is line count, 1 is label , 2 is value
            extracted_data.append([count, entity.label_, entity.text])
    return extracted_data



def returnStructuredData(unstructureddata):
    extracted_data_custom = extract(unstructureddata)
    return extracted_data_custom