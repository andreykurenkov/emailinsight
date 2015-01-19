import mailbox
import os
import re

class parsedEmail():

    def __init__(self,label,subject,sender,fromDomain,timeRec,words):
        self.label = label
        self.subject = subject
        self.sender = sender
        self.fromDomain = fromDomain
        self.day = timeRec[0]
        self.date = timeRec[1]
        self.month = timeRec[2]
        self.year = timeRec[3]
        self.hour = timeRec[4]
        self.words = words

evilSubstringsRegex = ['<html>.*</html',\
                       '=20(.*\n)*=20',\
                       '\<.*\>',\
                        '\>.*\n',\
                        'Content\-.*[ \n]']
def cleanEmail(string):
    for evilRegex in evilSubstringsRegex:
        string = re.sub(evilRegex,'',string)
    return string

def addToCountDict(word,countDict):
    if word in countDict:
        countDict[word]+=1
    else:
        countDict[word]=1

def parseEmails(folder,printInfo=True):
    files = os.listdir(folder)
    emails = []
    for aFile in files:
        if os.path.isdir(aFile):
            emails += parseEmails(folder+'/'+aFile)
        elif '.mbox' in aFile:
            category = (folder+'-'+aFile[:aFile.index('.')])[2:]
            if printInfo:
                print 'Parsing %s'%category
            box = mailbox.mbox(folder+'/'+aFile)
            for message in box:
                subject = message['subject']
                sender = re.sub('[\-=|;\"\>\<\'\)\(,.?!\n\r\t]','',message['from'])
                if '@' not in message['from']:
                    continue
                senderDomain = sender[sender.index('@'):]
                
                date = message['Date']
                dateParts = date.split(" ")
                dateParts[0] = dateParts[0][:-1]
                dateParts[4] = dateParts[4][:2]
                
                payload = message.get_payload()
                if message.is_multipart():
                    messageContent = payload[0].as_string()
                else:
                    messageContent = payload
                messageContent = cleanEmail(messageContent)
                if len(messageContent)>10000:
                    continue

                messageWords = messageContent.split(" ")
                
                wordCount = {}
                for word in messageWords:
                    if word=='-----Original Message-----':
                        break
                    if len(word)==0 or '\r' in word or '=' in word \
                       or '#' in word or '&' in word or word[0].isupper():
                        continue
                    word = re.sub('[|;\"\'\>\<\'\)\(,.?!\n]','',word)
                    if len(word)>3:
                        addToCountDict(word,wordCount)
                
                email = parsedEmail(category,subject,sender,senderDomain,\
                                                     dateParts,wordCount)
                emails.append(email)
            if printInfo:
                print 'Parsed %d emails\n'%len(box)
    return emails

def getEmailStats(emails):
    fromCount = {}
    domainCount = {}
    totalWordsCount = {}
    labels = []
    for email in emails:
        addToCountDict(email.sender,fromCount)
        addToCountDict(email.fromDomain,domainCount)
        if email.label not in labels:
            totalWordsCount[email.label] = {}
            labels.append(email.label)
    
        words = email.words
        for word in words.keys():
            if word not in totalWordsCount and word not in labels:
                totalWordsCount[word]=0
            if word not in totalWordsCount[email.label]:
                totalWordsCount[email.label][word]=0
            totalWordsCount[word]+=words[word]
            totalWordsCount[email.label][word]+=words[word]
    return (totalWordsCount,fromCount,domainCount,labels)

def getTopEmailCounts(emails,percentThresh=0.25,numWords=50,numSenders=20,numDomains=5):
    (totalWordsCount,fromCount,domainCount,labels) = getEmailStats(emails)
    topWords = set([])
    for label in labels:
        labelWords = totalWordsCount[label]
        topWordsDict = {}
        for word in labelWords:
            if word not in labels and labelWords[word]>totalWordsCount[word]*percentThresh:
                topWordsDict[word] = labelWords[word]
        sortedWords = sorted(topWordsDict,key=topWordsDict.get,reverse=True)
        for i in range(min(numWords,len(sortedWords))):
            topWords.add(sortedWords[i])

    sortedSenders = sorted(fromCount,key=fromCount.get,reverse=True)
    sortedDomains = sorted(domainCount,key=domainCount.get,reverse=True)
    print '%d words found.'%len(totalWordsCount)

    topSenders = sortedSenders[:numSenders]
    topDomains = sortedDomains if len(sortedDomains)<=numDomains else sortedDomains[:numDomains]
    return (topWords,topSenders,topDomains)

def mboxToBinaryCVS(folder):
    outputFile = open(folder+'/binaryEmails.csv','w')

    emails = parseEmails(folder)
    (topWords,topSenders,topDomains)=getTopEmailCounts(emails)

    for sender in topSenders:
        outputFile.write('Sender %s,'%sender)
    for domain in topDomains:
        outputFile.write('From domain %s,'%domain)
    for word in topWords:
        outputFile.write('Has %s,'%word)
    outputFile.write('label\n')

    for email in emails:
        for sender in topSenders:
            outputFile.write('1, ' if email.sender==sender else '0,')
        for domain in topDomains:
            outputFile.write('1, ' if email.fromDomain==domain else '0,')
        for word in topWords:
            outputFile.write('1, ' if word in email.words else '0,')
        outputFile.write(email.label+'\n')

def mboxToCVS(folder, name='email.csv', limitSenders=True, limitDomains=True):
    outputFile = open(folder+'/'+name,'w')

    emails = parseEmails(folder)
    (topWords,topSenders,topDomains)=getTopEmailCounts(emails)
    outputFile.write('Sender,Domain,')
    for word in topWords:
        outputFile.write('Has %s,'%word)
    outputFile.write('label\n')

    for email in emails:
        if not limitSenders or email.sender in topSenders:
            outputFile.write('%s,'%(email.sender))
        else:    
            outputFile.write('UncommonSender,')
        if not limitDomains or email.fromDomain in topDomains:
            outputFile.write('%s,'%(email.fromDomain))
        else:
            outputFile.write('UncommonDomain,')
        for word in topWords:
            outputFile.write('Yes,' if word in email.words else 'No,')
        outputFile.write(email.label+'\n')

folder = "."
mboxToCVS(folder,'limitedEmails.csv')
