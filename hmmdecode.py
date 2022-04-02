#hmm part of speech tagger

import sys, glob
import numpy as np

emissions = {}
transitions = {}
sorted_tags = []
isTransition = False
isEmission = False
isFreq = False

def viterbi(T, N, obs, write_file):

    #intialization step
    vit = np.zeros((N,T))
    backpointer = np.zeros((N,T), dtype=int)

    seen = False
    for tag in start_transitions:
        if obs[0].lower() in emissions[tag]:
            seen = True
            break
    
    for s,q in enumerate(start_transitions):
        #if the first word is seen
        if seen:

            #if emission for this state is 0 -> prob = 0
            if obs[0].lower() not in emissions[q]:
                vit[s,0] = 0

            #otherwise do as normal 
            else:
                vit[s,0] = start_transitions[q]*emissions[q][obs[0].lower()]
        else:
            #use start transitions if word has never been seen
            vit[s,0] = start_transitions[q]
    
    #main recursion
    for t in range(1,T):
        beenSeen = False
        for tag in transitions:
            if obs[t].lower() in emissions[tag]:
                beenSeen = True
                break
        #if word is not seen, use transition probabilities
        if not beenSeen:
            for s,q in enumerate(transitions):
                tmp = []
                for s1, q1 in enumerate(transitions):
                    #do a transition to an open class tag
                    for tag in sorted_tags:
                        if tag in transitions[q1]:
                            tmp.append(vit[s1, t-1]*transitions[q1][tag])
                            break
                tmp = np.array(tmp)
                vit[s,t] = np.amax(tmp)
                backpointer[s,t] = np.argmax(tmp)
        else:
            for s,q in enumerate(transitions):

                #if emission for this state is 0 -> prob is 0
                if obs[t].lower() not in emissions[q]:
                    vit[s,t] = 0
                    backpointer[s,t] = 100000

                #otherwise do as normal
                else:
                    tmp = []
                    for s1,q1 in enumerate(transitions):
                        #check if transition doesnt exist
                        if q not in transitions[q1]:
                            #do a transition to an open class tag
                            for tag in sorted_tags:
                                if tag in transitions[q1]:
                                    tmp.append(vit[s1, t-1]*transitions[q1][tag])
                                    break
                        else:
                            tmp.append(vit[s1, t-1]*transitions[q1][q])
                    tmp = np.array(tmp)
                    vit[s,t] = np.amax(tmp)*emissions[q][obs[t].lower()]
                    backpointer[s,t] = np.argmax(tmp)
    
    #best path
    bestpathpointer = np.argmax(vit[:,T-1])
    path_states = []
    path_states.append(bestpathpointer)
    t = T-1
    while t > 0:
        path_states.append(backpointer[bestpathpointer,t])
        bestpathpointer = backpointer[bestpathpointer,t]
        t -= 1
    path_states.reverse()
    states = list(transitions)
    for word in zip(obs, path_states):
        write_file.write(word[0] + "/" + states[word[1]].upper() + " ")
    write_file.write("\n")

model_file = glob.glob("hmmmodel.txt")
with open(model_file[0], 'r') as file:
    currTag = ""
    for line in file:
        data = line.split()
        if data[0] == "transitions:":
            isTransition = True
            continue
        elif data[0] == "emissions:":
            isEmission = True
            isTransition = False
            continue
        elif data[0] == "tagfreq:":
            isFreq = True
            isEmission = False
            continue
        if isTransition and not isEmission:
            if len(data) == 1:
                currTag = data[0]
                transitions[currTag] = {}
            else:
                transitions[currTag][data[0]] = float(data[1])
        elif isEmission and not isTransition:
            if len(data) == 1:
                currTag = data[0]
                emissions[currTag] = {}
            else:
                emissions[currTag][data[0]] = float(data[1])
        elif isFreq:
            sorted_tags.append(data[0])

start_transitions = transitions['Sentence_Start']
transitions.pop("Sentence_Start")

with open(sys.argv[1], 'r') as file:
    write_file = open("hmmoutput.txt", 'w')
    for line in file:
        line = line.split()
        viterbi(len(line), len(transitions), line, write_file)

