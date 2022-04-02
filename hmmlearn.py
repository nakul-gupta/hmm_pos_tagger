#hmm part of speech tagger

import sys

tag_freq = {}
tag_vocab_count = {}
emissions = {}
transitions = {}

with open(sys.argv[1], 'r') as file:
    for line in file:
        prev_tag = "Sentence_Start"
        for word in line.split():
            symbols = word.split("/")
            new_word = symbols[0].lower()
            tag = symbols[len(symbols)-1].lower()

            #weird edge case
            if tag == "":
                new_word = "/"

            #calculate tag baseline freq
            if tag not in tag_freq:
                tag_freq[tag] = 1
            else:
                tag_freq[tag] += 1

            #calculate tag baseline freq for starting state if new sentence
            if prev_tag == "Sentence_Start":
                if prev_tag not in tag_freq:
                    tag_freq[prev_tag] = 1
                else:
                    tag_freq[prev_tag] += 1

            #calculate the freq of the tag being associated with this word
            if tag not in emissions:
                emissions[tag] = {}
                emissions[tag][new_word] = 1
            elif new_word not in emissions[tag]:
                emissions[tag][new_word] = 1
            else:
                emissions[tag][new_word] += 1
            
            #calculate the freq of this tag following the previous tag
            if prev_tag not in transitions:
                transitions[prev_tag] = {}
                transitions[prev_tag][tag] = 1
            elif tag not in transitions[prev_tag]:
                transitions[prev_tag][tag] = 1
            else:
                transitions[prev_tag][tag] += 1

            #count tag vocab
            if tag not in tag_vocab_count:
                tag_vocab_count[tag] = []
            if word not in tag_vocab_count[tag]:
                tag_vocab_count[tag].append(word)
            
            #set this tag to previous tag
            prev_tag = tag

#calculate transition probabilities
for tag in transitions:
    for transition_tag in transitions[tag]:
        transitions[tag][transition_tag] /= tag_freq[tag]

#calculate emissions probabilities
for tag in emissions:
    for word in emissions[tag]:
        emissions[tag][word] /= tag_freq[tag]

#calculate tag vocab count
for tag in tag_vocab_count:
    count = len(tag_vocab_count[tag])
    tag_vocab_count[tag] = count
sorted_tags = sorted(tag_vocab_count, key=tag_vocab_count.get)

#print model
file = open('hmmmodel.txt','w')
file.write("transitions:\n")
for tag in transitions:
    file.write(tag + "\n")
    for transition_tag in transitions[tag]:
        file.write("\t" + transition_tag + " " + str(transitions[tag][transition_tag]) + "\n")
file.write("emissions:\n")
for tag in emissions:
    file.write(tag + "\n")
    for word in emissions[tag]:
        file.write("\t" + word + " " + str(emissions[tag][word]) + "\n")
file.write("tagfreq:\n")
for tag in sorted_tags:
    file.write(tag + "\n")
