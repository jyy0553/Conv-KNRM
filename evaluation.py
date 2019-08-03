from __future__ import division
import pandas as pd 
import subprocess
import platform,os
import sklearn
import numpy as np
import configure

# def mrr_metric(group):
# 	group = sklearn.utils.shuffle(group,random_state =132)
# 	candidates=group.sort_values(by='score',ascending=False).reset_index()
# 	rr=candidates[candidates["flag"]>0].index.min()+1
# 	# print ("rr :"+str(rr))
# 	if rr!=rr:
# 		return 0
# 	return 1.0/rr

# def map_metric(group):
# 	group = sklearn.utils.shuffle(group,random_state =132)
# 	ap=0
# 	candidates=group.sort_values(by='score',ascending=False).reset_index()
# 	correct_candidates=candidates[candidates["flag"]>0]
# 	if len(correct_candidates)==0:
# 		return 0
# 	for i,index in enumerate(correct_candidates.index):
# 		ap+=1.0* (i+1) /(index+1)   
# 	#print( ap/len(correct_candidates))
# 	return ap/len(correct_candidates)

FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()

#获取query的相关文档个数
def get_query_qrels_number(qid):
	R = 0
	#web trec 2009-2011
	file = "../pre-process/"+FLAGS.data+"/qrels_1-150.txt"
	# file = "../qrels_1-150.txt"
	f = open(file)
	for line in f.readlines():
		lineList = line.split(" ")
		# print (lineList[0] in qid)
		if int(lineList[0])<=50 and int(lineList[0]) == qid:
			if int(lineList[2])>0:
				R+=1
		elif int(lineList[0])>50 and int(lineList[3])>0 and int(lineList[0]) == qid:
			R+=1
		# if int(lineList[0]) == qid and int(lineList[2])>0:
		# 	R+=1
	# print ("R "+str(R))
	f.close()
	return R
# def get_query_qrels_number(qid):
# 	R = 0
# 	#web trec 2009-2011
# 	# file = "../data_set/clueweb09B-title/prels.catB.1-50"
# 	file =  "../qrels_1-150.txt"
# 	f = open(file)
# 	for line in f.readlines():
# 		lineList = line.split(" ")
# 		# print (lineList[0] in qid)
# 		if int(lineList[0]) == qid and int(lineList[2])>0:
# 			R+=1
# 	# print ("R "+str(R))
# 	f.close()
# 	return R

# for i in range(1,51):
# 	a = get_query_qrels_number(i)
# 	print ("query %s have %s corr "%(i,a))

def map_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	ap=0
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	qid = correct_candidates["query_ID"][0]
	R = get_query_qrels_number(qid)
	if R == 0:
		R = 1
	rank_list = correct_candidates["flag"]	
	is_one = 1
	index = 1
	if len(correct_candidates)==0:
		return 0
	for i in rank_list:
		if i > 0:
			ap += (1.0*is_one)/(1.0*index)
			is_one += 1
		index += 1
	return ap/(1.0*R)


def NDCG_metric(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	group1 = sklearn.utils.shuffle(group,random_state =132)
	AtP=20
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	if len(correct_candidates) == 0:
		return 0
	rank_list = correct_candidates["flag"]
	# print ("ndcg")
	# print (rank_list)
	p = 1
	DCG = 0
	index = 1
	for i in rank_list:
		if p >AtP:
			break
		# DCG += (2**i-1)/np.log(index+1)
		DCG += (np.power(2.0,i)-1.0)/np.log2(index+1)
		index += 1
		p += 1

	candidates1 =group1.sort_values(by='flag',ascending=False)
	rank_list1 = candidates1["flag"]
	# print(rank_list1)

	p = 1
	IDCG = 0
	index = 1
	for i in rank_list1:
		if p > AtP:
			break
		# IDCG += (2**i-1)/np.log(index+1)
		IDCG += (np.power(2.0,i)-1.0)/np.log2(index+1)
		index += 1
		p += 1

	if IDCG == 0:
		IDCG += 0.00001

	nDCG = float(DCG/IDCG)
	# print ("ndcg: "+str(nDCG))
	return nDCG

def NDCG_metric1(group):
	group = sklearn.utils.shuffle(group,random_state =132)
	group1 = sklearn.utils.shuffle(group,random_state =132)
	AtP=20
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	if len(correct_candidates)==0:
		return 0
	rank_list = correct_candidates["flag"]
	# print ("ndcg")
	# print (rank_list)

	p = 1
	DCG = 0
	index = 1

	for i in rank_list:
		if p >AtP:
			break
		if index == 1:
			DCG += i
			index += 1
			p += 1
			continue		
		# DCG += (2**i-1)/np.log(index+1)
		DCG += (1.0*i)/np.log2(index)
		index += 1
		p += 1

	candidates1 =group1.sort_values(by='flag',ascending=False)
	rank_list1 = candidates1["flag"]
	# print(rank_list1)

	p = 1
	IDCG = 0
	index = 1
	for i in rank_list1:
		if p > AtP:
			break
		if index == 1:
			IDCG += i
			index += 1
			p += 1
			continue
		IDCG += (1.0*i)/np.log2(index)
		index += 1
		p += 1
	if IDCG == 0:
		IDCG += 0.00001
	nDCG = float(DCG/IDCG)
	# print ("ndcg: "+str(nDCG))
	return nDCG


def ERR_metric(group):
	AtP = 20

	if len(group) <AtP:
		AtP = len(group)

	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	if len(correct_candidates)==0:
		return 0
	rank_list = correct_candidates["flag"]
	gmax = rank_list.max()
	# gmax = 2
	ERR = 0
	# r = 1
	# for i in rank_list:
	# 	ppr = 1
	# 	if r > AtP:
	# 		break
	# 	for j in range(1,r):
	# 		# Rj = (2**rank_list[j]-1)/2**gmax
	# 		Rj = (np.power(2,rank_list[j])-1.0)/np.power(2,gmax)
	# 		ppr *= (1-Rj)
	# 		# p += 1
	# 	# Rr = (2**i-1)/2**gmax
	# 	Rr = (np.power(2,i)-1.0)/np.power(2,gmax)
	# 	ppr *= Rr
	# 	ERR += (1/(np.log(r+1)))*ppr
	# 	r += 1
	for r in range(1,AtP+1):
		pp_r = 1
		for i in range(1,r):
			R_i = float((np.power(2.0,rank_list[i-1])-1.0)/np.power(2.0,gmax))
			pp_r *= (1.0 - R_i)
		R_r = float((np.power(2.0,rank_list[r-1])-1.0)/np.power(2.0,gmax))
		pp_r *= R_r
		ERR += (1.0/r)*pp_r 

	return ERR

def p_metric(group):
	AtP = 20
	group = sklearn.utils.shuffle(group,random_state =132)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]>=-15]
	rank_list = correct_candidates["flag"]
	true_num = 0
	false_num = 0
	index = 1
	for i in rank_list:
		if index > AtP:
			break
		if i>0:
			true_num+=1
		else:
			false_num+=1
		index+=1
	# p = true_num/(true_num+false_num)
	p = float((1.0*true_num)/(1.0*AtP))
	# print (rank_list)
	# print (p)
	return p

def evaluationBypandas(df,predicted):
	df["score"]=predicted
	map= df.groupby("query_content").apply(map_metric).mean()
	NDCG0 = df.groupby("query_content").apply(NDCG_metric).mean()
	# print ("NDCG1 start")
	NDCG1 = df.groupby("query_content").apply(NDCG_metric1).mean()
	# print ("err start")
	ERR = df.groupby("query_content").apply(ERR_metric).mean()
	# print ("p start")
	p = df.groupby("query_content").apply(p_metric).mean()
	# print ("NDCGAT3: "+str(NDCGAT3))
	# print ("ERR: "+str(ERR))
	# print ("p: "+str(p))
	return map,NDCG0,NDCG1,ERR,p



# from pre_dateset import load,creat_train_test,get_wordDic_Embedding,batch_gen_with_single,batch_gen_with_point_wise,get_overlap_dict
# from IR_train_test_1-150_1 import predict
# # creat_train_test("2")
# # train,test = load()
# # map_mrr_test = evaluation.evaluationBypandas(test,predicted_test[:,-1])
# # precision_test = evaluation.precision(test,predicted_test[:,-1])
# predicted = "0.36918074 0.3609913  0.4322757  0.3593319  0.5214317  0.37470037  0.3525386  0.4022195  0.38703036 0.30913353 0.32793254 0.6493802  0.34465486 0.38864785 0.35372138 0.40301096 0.32749617 0.38914144  0.36982882 0.40039355 0.41679916 0.33136183 0.37340838 0.3686685  0.36938745 0.5039977  0.45070666 0.31595045 0.31931144 0.4479711  0.37506217 0.45662773 0.3741472  0.38564312 0.35232732 0.38787913  0.29477128 0.4404712  0.3369081  0.37143946 0.5068547  0.49271572  0.2964103  0.33724788 0.421629   0.3282137  0.4453522  0.29997283  0.36435473 0.29412222 0.377243   0.41567585 0.41118893 0.3517627  0.39116645 0.5068547  0.37347892 0.33931744 0.628632   0.31527877  0.34039223 0.32961744 0.38404265 0.33666158 0.3482583  0.39487106  0.39390677 0.4133073  0.41024804 0.43463618 0.40137935 0.46609735   0.3579919  0.30566874 0.4161304  0.4057867  0.34298044 0.31041822  0.41828242 0.3968929  0.41052413 0.53703755 0.39725605 0.4659115  0.36660105 0.41758153 0.41008496 0.35429245 0.42477906 0.33892426  0.37426415 0.3587442  0.3641602  0.37736624 0.44360256 0.34908354  0.41691032 0.49365023 0.39391553 0.5030624  0.2941833  0.29855478   0.430027   0.36129457 0.44008783 0.38243878 0.3787488  0.40959132 0.37634802 0.38108605 0.50786984 0.3575533  0.41052413 0.29322594  0.33691013 0.37848878 0.340904   0.3503558  0.30543542 0.36571664  0.36158895 0.44455996 0.30831462 0.37303537 0.35200912 0.36670083  0.38348007 0.55218506 0.33476806 0.36036003 0.4567692  0.37377802  0.30903107 0.3205703  0.38008022 0.40862405 0.38413742 0.36543602  0.471836   0.36995435 0.6290537  0.30487838 0.43707862 0.40797934   0.35587573 0.34393555 0.3513232  0.44239265 0.37094128 0.38485336 0.37513983 0.3446949  0.4275074  0.3231458  0.4702105  0.37534985  0.34706926 0.3331123  0.4313696  0.3947987  0.3207873  0.46106753 0.42039046 0.44478384 0.65443265 0.49677837 0.39868787 0.37832886  0.38775712 0.39930877 0.6544273  0.3553679  0.4565607  0.311171  0.34539542 0.4313083  0.41644987 0.40221488 0.54160786 0.40355006  0.3513232  0.40139797 0.3220484  0.3317653  0.5289874  0.3388219  0.2957111  0.4652078  0.44103962 0.4148218  0.3998869  0.2886181  0.38654682 0.34590727 0.36889917 0.5208007  0.31446725 0.4091493  0.31966358 0.38718104 0.29689795 0.41578156 0.6731894  0.38375962  0.29163718 0.44343066 0.4328186  0.34452435 0.33540726 0.31215718  0.40305477 0.40589502 0.3741472  0.3753748  0.40039355 0.46422696  0.38468903 0.4352971  0.40959227 0.45478866 0.310417   0.4313083  0.37755263 0.36402315 0.3502139  0.43123862 0.42810854 0.37244633  0.34590727 0.48285133 0.37069827 0.3741938  0.3806966  0.52033925  0.44310972 0.32444608 0.2977664  0.50786984 0.43864998 0.37874293  0.36310753 0.38514286 0.36499774 0.4030673  0.3470512  0.32843393  0.34715587 0.3838328  0.33043194 0.4185641  0.35600358 0.44365847  0.2970013  0.3562524  0.37027508 0.46422502 0.34474158 0.3566832  0.3349464  0.30884784 0.44195056 0.4261989  0.31962293 0.33586246  0.38580847 0.3509282  0.6181122  0.40179902 0.39004803 0.46146497  0.34954935 0.30684584 0.46591043 0.33777988 0.3328183  0.399804  0.29217643 0.5031291  0.36613065 0.30512595 0.31400716 0.39989838  0.37917984 0.5443732  0.41409618 0.37414557 0.41576728 0.44620982  0.33814406 0.31306386 0.45102444 0.3758145  0.40268984 0.6544273  0.30306864 0.45947543 0.3729394  0.31850561 0.49467087 0.34369665  0.3457383  0.50119716 0.4289446  0.33746362 0.3606934  0.28548735  0.35327834 0.37442964 0.3701777  0.38449758 0.3465942  0.35070354  0.3315292  0.48071384 0.44180924 0.42816556 0.3515396  0.37671816  0.34590727 0.5565846  0.46741676 0.49872708 0.3685609  0.66933286  0.3050995  0.34098315 0.65443265 0.3295552  0.33319014 0.49816346  0.35321474 0.38454932 0.31101257 0.29163718 0.3610934  0.3972891  0.36224335 0.44243205 0.4634992  0.45832652 0.36199296 0.40887564  0.34725058 0.4651996  0.31535408 0.45065042 0.4126135  0.43475845  0.40369692 0.3500842  0.40639305 0.38748506 0.33661342 0.36459795  0.340904   0.33600038 0.54126215 0.33319014 0.34302497 0.3743599  0.49468037 0.35847604 0.33558738 0.3660218  0.367944   0.377243  0.3062996  0.3823117  0.3777533  0.32465714 0.40974614 0.40035298  0.30403087 0.411167   0.34185508 0.41071782 0.4095244  0.31535113  0.36543456 0.34590727 0.5099889  0.39890474 0.3998083  0.34006265 0.3845948  0.41442415 0.50786984 0.3642804  0.34834272 0.6544273  0.45230567 0.4768161  0.34691602 0.36137086 0.41162315 0.34024942  0.32074738 0.2904116  0.33549917 0.5457387  0.316342   0.29739302  0.340864   0.30643272 0.37738824 0.43318236 0.4063941  0.45595282  0.3648402  0.38444906 0.33967042 0.34225774 0.34827352 0.34095258 0.50786984 0.42199475 0.33976036 0.41320577 0.43451187 0.53808886 0.40512443 0.31933197 0.42249036 0.34058172 0.40131596 0.3800984 0.38050556 0.43339193 0.33738005 0.44580075 0.47064832 0.39790654 0.32187712 0.44008455 0.38218582 0.40132144 0.37659597 0.36340085 0.36233693 0.32430843 0.3296169  0.49459976 0.6398194  0.40200272 0.41146332 0.5341139  0.56319964 0.4195585  0.37262297 0.5347264 0.4005384  0.39620182 0.34498024 0.42846388 0.28557432 0.37309536 0.52297544 0.34317338 0.37391397 0.6544273  0.39004803 0.3316516 0.34706604 0.3332395  0.37856323 0.33481127 0.49468037 0.3115489 0.37274942 0.36007532 0.4003403  0.3947987  0.38108605 0.33364403 0.30601117 0.5031228  0.2870344  0.40717417 0.4063882  0.36348596 0.29240608 0.37717086 0.4040069  0.39028823 0.45913953 0.36867896 0.35314375 0.6544273  0.43899462 0.33955473 0.29576433 0.3597963 0.3974018  0.35880324 0.40478158 0.37862325 0.3276332  0.32615283 0.41283903 0.387897   0.30432704 0.37072816 0.413325   0.31478333 0.3364637  0.305857   0.40919176 0.377243   0.38060826 0.31478336 0.42630988 0.3922075  0.35201597 0.3272146  0.39404327 0.38839808 0.37561888 0.4243205  0.36348617 0.36754984 0.4464987  0.39930877 0.3118936  0.6724541  0.43828577 0.28261364 0.3741938  0.4393459 0.3498301  0.40851632 0.49954748 0.39032423 0.38770783 0.43561837 0.39937058 0.3949365  0.35322696 0.5186665  0.38606304 0.39771676 0.30333954 0.43339193 0.35263354 0.67703366 0.41162315 0.49467087 0.4108626  0.35283554 0.30913353 0.38914806 0.4683642  0.46422696 0.43194026 0.3816639  0.3494693  0.44160983 0.45270142 0.32341313 0.42723566 0.42654753 0.39240456 0.44595656 0.40354943 0.32840595 0.38087296 0.28517815 0.29193217 0.37875396 0.4801823  0.41803172 0.35175276 0.38512877 0.40223566 0.3312716  0.44455996 0.43487418 0.37813634 0.3498519  0.47017708 0.34659708 0.34121662 0.30306864 0.28824818 0.54746044 0.44243532 0.34797698 0.36487716 0.30940756 0.5086008  0.377243   0.40920097 0.34948492 0.3648712  0.3844156 0.40977594 0.49471584 0.39186764 0.36257586 0.4002389  0.41052592 0.4303465  0.5034108  0.3513235  0.32240415 0.30942196 0.5630623 0.46460304 0.39321783 0.37764674 0.28744292 0.30664736 0.37026307 0.40081072 0.37982923 0.39974067 0.3583958  0.34465477 0.3498145 0.34294945 0.3461822  0.29557967 0.29532757 0.40585738 0.3770675 0.30969566 0.3939572  0.350545   0.4506164  0.30083346 0.42685863 0.30042374 0.3751176  0.3583695  0.401107   0.39465815 0.35894215 0.39576197 0.37927556 0.40221488 0.34803677 0.3190941  0.42106858 0.43031892 0.39972848 0.33814156 0.3427831  0.50297594 0.53941464"
# # print (np.array(predicted))
# predicted_test = np.matrix(predicted).T
# # Y_predict = np.squeeze(d['predicted_test'])
# train,test = load()
# # print (test["flag"])
# # print (len(predicted_test))
# # print (predicted_test[:,-1])
# map_mrr_NDCG_test = evaluationBypandas(test,predicted_test[:,-1])
# print (map_mrr_NDCG_test)
# precision_test = precision(test,predicted_test[:,-1])
