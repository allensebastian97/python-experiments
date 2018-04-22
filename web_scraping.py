from bs4 import BeautifulSoup

import subprocess, sys, urllib2, request, re



url = "http://business.inquirer.net/232452/peso-slides-nearly-11-year-low-level"
url =  "http://hindi.webdunia.com"
url = "http://hindi.webdunia.com/international-hindi-news/suicide-attack-at-iraq-displaced-camp-117070300009_1.html"

#r  = requests.get(url)

p = subprocess.Popen("curl " + url, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
(output, err) = p.communicate()

#print output


matchObj = re.search('.*(outbrain|taboola).*', output, re.I)
if matchObj:
	mat=matchObj.group(1)

	if mat == 'taboola' :
		taboola = 1
	else if mat == 'outbrain' :
		outbrain = 1
		
		outbrain = re.search('.*(outbrain).*', mat, re.I)

else: 
	link=re.search('.*href=["\'](.*)[\'"].*',output, re.I)
	print link.group(1)

sys.exit()


