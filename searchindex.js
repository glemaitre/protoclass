Search.setIndex({envversion:49,filenames:["api","generated/protoclass.data_management.ADCModality","generated/protoclass.data_management.DCEModality","generated/protoclass.data_management.DWIModality","generated/protoclass.data_management.GTModality","generated/protoclass.data_management.OCTModality","generated/protoclass.data_management.T2WModality","generated/protoclass.extraction.BrixQuantificationExtraction","generated/protoclass.extraction.DCTExtraction","generated/protoclass.extraction.EdgeSignalExtraction","generated/protoclass.extraction.EnhancementSignalExtraction","generated/protoclass.extraction.GaborBankExtraction","generated/protoclass.extraction.HaralickExtraction","generated/protoclass.extraction.IntensitySignalExtraction","generated/protoclass.extraction.PUNQuantificationExtraction","generated/protoclass.extraction.PhaseCongruencyExtraction","generated/protoclass.extraction.SemiQuantificationExtraction","generated/protoclass.extraction.SpatialExtraction","generated/protoclass.extraction.ToftsQuantificationExtraction","generated/protoclass.extraction.WeibullQuantificationExtraction","generated/protoclass.utils.check_filename_pickle_load","generated/protoclass.utils.check_filename_pickle_save","generated/protoclass.utils.check_img_filename","generated/protoclass.utils.check_modality","generated/protoclass.utils.check_modality_gt","generated/protoclass.utils.check_modality_inherit","generated/protoclass.utils.check_npy_filename","generated/protoclass.utils.check_path_data","generated/protoclass.utils.find_nearest","index","install","support"],objects:{"protoclass.data_management":{ADCModality:[1,1,1,""],DCEModality:[2,1,1,""],DWIModality:[3,1,1,""],GTModality:[4,1,1,""],OCTModality:[5,1,1,""],T2WModality:[6,1,1,""]},"protoclass.data_management.ADCModality":{__init__:[1,2,1,""],get_pdf:[1,2,1,""],is_read:[1,2,1,""],read_data_from_path:[1,2,1,""],update_histogram:[1,2,1,""]},"protoclass.data_management.DCEModality":{__init__:[2,2,1,""],build_heatmap:[2,2,1,""],get_pdf_list:[2,2,1,""],is_read:[2,2,1,""],read_data_from_path:[2,2,1,""],update_histogram:[2,2,1,""]},"protoclass.data_management.DWIModality":{__init__:[3,2,1,""],is_read:[3,2,1,""],read_data_from_path:[3,2,1,""],update_histogram:[3,2,1,""]},"protoclass.data_management.GTModality":{__init__:[4,2,1,""],extract_gt_data:[4,2,1,""],is_read:[4,2,1,""],read_data_from_path:[4,2,1,""],update_histogram:[4,2,1,""]},"protoclass.data_management.OCTModality":{__init__:[5,2,1,""],is_read:[5,2,1,""],read_data_from_path:[5,2,1,""],update_histogram:[5,2,1,""]},"protoclass.data_management.T2WModality":{__init__:[6,2,1,""],get_pdf:[6,2,1,""],is_read:[6,2,1,""],read_data_from_path:[6,2,1,""],update_histogram:[6,2,1,""]},"protoclass.extraction":{BrixQuantificationExtraction:[7,1,1,""],DCTExtraction:[8,1,1,""],EdgeSignalExtraction:[9,1,1,""],EnhancementSignalExtraction:[10,1,1,""],GaborBankExtraction:[11,1,1,""],HaralickExtraction:[12,1,1,""],IntensitySignalExtraction:[13,1,1,""],PUNQuantificationExtraction:[14,1,1,""],PhaseCongruencyExtraction:[15,1,1,""],SemiQuantificationExtraction:[16,1,1,""],SpatialExtraction:[17,1,1,""],ToftsQuantificationExtraction:[18,1,1,""],WeibullQuantificationExtraction:[19,1,1,""]},"protoclass.extraction.BrixQuantificationExtraction":{__init__:[7,2,1,""],fit:[7,2,1,""],load_from_pickles:[7,2,1,""],save_to_pickles:[7,2,1,""],transform:[7,2,1,""]},"protoclass.extraction.DCTExtraction":{__init__:[8,2,1,""],fit:[8,2,1,""],load_from_pickles:[8,2,1,""],save_to_pickles:[8,2,1,""],transform:[8,2,1,""]},"protoclass.extraction.EdgeSignalExtraction":{__init__:[9,2,1,""],fit:[9,2,1,""],load_from_pickles:[9,2,1,""],save_to_pickles:[9,2,1,""],transform:[9,2,1,""]},"protoclass.extraction.EnhancementSignalExtraction":{__init__:[10,2,1,""],fit:[10,2,1,""],load_from_pickles:[10,2,1,""],save_to_pickles:[10,2,1,""],transform:[10,2,1,""]},"protoclass.extraction.GaborBankExtraction":{__init__:[11,2,1,""],fit:[11,2,1,""],load_from_pickles:[11,2,1,""],save_to_pickles:[11,2,1,""],transform:[11,2,1,""]},"protoclass.extraction.HaralickExtraction":{__init__:[12,2,1,""],fit:[12,2,1,""],load_from_pickles:[12,2,1,""],save_to_pickles:[12,2,1,""],transform:[12,2,1,""]},"protoclass.extraction.IntensitySignalExtraction":{__init__:[13,2,1,""],fit:[13,2,1,""],load_from_pickles:[13,2,1,""],save_to_pickles:[13,2,1,""],transform:[13,2,1,""]},"protoclass.extraction.PUNQuantificationExtraction":{__init__:[14,2,1,""],fit:[14,2,1,""],load_from_pickles:[14,2,1,""],save_to_pickles:[14,2,1,""],transform:[14,2,1,""]},"protoclass.extraction.PhaseCongruencyExtraction":{__init__:[15,2,1,""],fit:[15,2,1,""],load_from_pickles:[15,2,1,""],save_to_pickles:[15,2,1,""],transform:[15,2,1,""]},"protoclass.extraction.SemiQuantificationExtraction":{__init__:[16,2,1,""],fit:[16,2,1,""],load_from_pickles:[16,2,1,""],save_to_pickles:[16,2,1,""],transform:[16,2,1,""]},"protoclass.extraction.SpatialExtraction":{__init__:[17,2,1,""],fit:[17,2,1,""],load_from_pickles:[17,2,1,""],save_to_pickles:[17,2,1,""],transform:[17,2,1,""]},"protoclass.extraction.ToftsQuantificationExtraction":{__init__:[18,2,1,""],compute_aif:[18,3,1,""],conc_to_signal:[18,2,1,""],fit:[18,2,1,""],load_from_pickles:[18,2,1,""],population_based_aif:[18,3,1,""],save_to_pickles:[18,2,1,""],signal_to_conc:[18,2,1,""],transform:[18,2,1,""]},"protoclass.extraction.WeibullQuantificationExtraction":{__init__:[19,2,1,""],fit:[19,2,1,""],load_from_pickles:[19,2,1,""],save_to_pickles:[19,2,1,""],transform:[19,2,1,""]},"protoclass.utils":{check_filename_pickle_load:[20,4,1,""],check_filename_pickle_save:[21,4,1,""],check_img_filename:[22,4,1,""],check_modality:[23,4,1,""],check_modality_gt:[24,4,1,""],check_modality_inherit:[25,4,1,""],check_npy_filename:[26,4,1,""],check_path_data:[27,4,1,""],find_nearest:[28,4,1,""]},protoclass:{data_management:[0,0,0,"-"],extraction:[0,0,0,"-"],utils:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","staticmethod","Python static method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:staticmethod","4":"py:function"},terms:{"1st":18,"2nd":18,"case":18,"default":[1,2,3,4,5,6,7,8,9,12,15,17,18],"final":[18,30],"float":[1,2,3,5,6,18],"int":[1,2,3,4,5,6,8,12,18,28],"return":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],"static":18,"true":[1,2,3,4,5,6,18],"while":[1,2,6],__init__:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],accord:17,acquisit:2,adc:1,address:[18,31],addtional:0,affect:4,agent:18,aif:18,aif_param:18,all:[2,3,17],allow:0,almost:30,alpha:[11,18],also:2,amplitud:18,anaconda:30,angl:[11,18],ani:31,anoth:25,aorta:18,appear:18,appli:[7,8,9,10,11,12,13,14,15,16,17,18,19],approach:18,apt:30,area:18,argument:[4,12],arrai:[2,5,9,28],arteri:18,associ:[1,2,4,6,18,28],attribut:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],auc:16,auto:[1,2,3,6],avail:[0,9],averag:18,axi:18,azimuth:9,b_seq:3,bank:11,base:[7,8,9,10,11,12,13,14,15,16,17,18,19],base_mod:[7,8,9,10,11,12,13,14,15,16,17,18,19],base_modality_:[8,9,10,11,12,13,15,17],becaus:29,becom:18,been:[1,2,3,4,5,6,18,21],befor:[18,30],beta:[14,18],between:18,bia:18,bin:[1,2,3,5,6,12],bin_:[1,5,6],bin_data:[1,6],bin_list:2,bin_series_:[2,3],bins_heatmap:2,biologi:18,bit:[5,12],blood:18,bool:[1,2,3,4,5,6,18],bpp:[7,8,9,10,11,12,13,14,15,16,17,18,19],brix:7,bucklei:18,build_heatmap:2,buonaccorsi:18,c_t:18,call:[1,6],can:[1,2,4,5,6,9,15,17,18,30,31],cannot:[7,8,9,10,11,12,13,14,15,16,17,18,19],cat:[7,8,9,10,11,12,13,14,15,16,17,18,19,24],cat_gt:4,cat_gt_:4,categor:4,cb_t:18,cb_t_:18,center:18,centr:17,check:[20,21,22,23,24,25,26,27,31],cheung:18,child:25,child_mod:25,child_object:25,choic:4,circl:18,classif:0,classinfo:25,clone:30,cluster:18,code:[5,30],cogito:29,com:30,come:30,comput:[1,2,3,4,5,6,8,9,11,12,15,17,18],compute_aif:18,conc:18,conc_to_sign:18,concentr:18,conda:30,congruenc:15,consid:[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19],consist:24,constant:18,constructor:[2,3,4,5],contact:29,contain:[1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19],contan:15,content:29,continu:30,contour:17,contrast:18,contribut:29,cooccur:[8,12],coord_system:17,coordin:17,correspond:[4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24],could:7,count:12,coverag:29,cp_t_:18,cylindr:17,cython:30,dale:18,data_:[1,2,3,4,5,6,9,17],data_manag:0,davi:18,dce:[2,3,18],dce_mod:18,dct:8,decai:18,deg:18,delai:18,denni:18,depend:[4,30],deriv:18,desir:[4,17],detect:18,determin:18,dev:30,deviat:11,diamet:18,dict:[1,2,6,15],dict_param:15,dictionari:[2,15],dictionnari:[1,6],differ:[1,2,3,4,5,6,11,15,30],dimens:5,direct:11,distanc:[12,17,18],document:0,done:18,dtype:[5,28],duerk:18,dure:0,dwi:3,dynam:18,each:[1,2,3,4,5,6,11,18],eccentr:18,echo:18,edge:[9,11,15],edge_detector:9,effect:18,eigen3:30,either:[15,17,18],element:[1,2,6],elev:9,ellips:18,enhanc:[16,18],enhancement:[7,10,14,16,18,19],ensur:30,equal:[7,8,9,10,11,12,13,14,15,16,17,18,19],equival:2,ergo:29,error:[25,29],estim:18,euclidean:17,exist:27,experiment:18,exponenti:18,extend:18,extens:[7,8,9,10,11,12,13,14,15,16,17,18,19,21],extract:[0,1,2,4,6],extract_gt_data:4,fals:18,featur:[0,7,8,9,10,11,12,13,14,15,16,17,18,19],file:[5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,26],filenam:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,26],filer:11,filter:[9,11,15],find:[7,10,13,14,16,18,19,28],first:[1,2,3,5,6],fit:[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19],fit_aif:18,fix:18,flash:18,flexibl:[7,8,9,10,11,12,13,14,15,16,17,18,19],flip:18,flip_angl:18,flow:29,focal:18,folder:[1,3,4,6],follow:[0,7,14,16,18,19,31],form:18,format:0,found:[1,2,3,5,6,18,28],frequenc:11,from:[1,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,25,30],full:[0,4,30],gabor:11,gabor_filter_3d:0,gamma:11,gaussian:18,gener:[5,18],get:29,get_pdf:[1,6],get_pdf_list:2,git:30,github:[30,31],given:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],glemaitr:30,global:18,gradient:[9,18],greater:18,grei:12,ground:[4,7,8,9,10,11,12,13,14,15,16,17,18,19,24],ground_truth:[7,8,9,10,11,12,13,14,15,16,17,18,19,24],guassian:18,guillaum:31,handl:[0,1,2,3,4,5,6],haralick:12,have:[1,2,3,4,5,6,18],heatmap:2,hematocrit:18,hematocrit_:18,hematrocrit:18,high:18,higher:12,hillenbrand:18,histogram:[1,2,3,4,5,6],hoffmann:7,http:30,human:29,idx:28,iii:9,imag:[1,2,3,4,5,6,8,9,11,12,15,17,30],image:18,img:[5,22],includ:[0,18],index:[4,7,8,9,10,11,12,13,14,15,16,17,18,19,24,28,29],indic:12,indice:[1,2,6],infer:18,info:25,inform:[1,2,4,6,17],inher:[7,8,9,10,11,12,13,14,15,16,17,18,19],inherit:25,initi:18,inject:18,input:[12,18],insid:[2,28],instal:30,install:29,insterest:24,integ:[1,2,3,5,6,12,18],integr:30,intens:[1,2,3,5,6],intensiti:13,intensity_rang:2,interest:[7,8,9,10,11,12,13,14,15,16,17,18,19,23,24],interv:18,is_read:[1,2,3,4,5,6],issu:31,jackson:18,jayson:18,jesberg:18,journal:18,kel:7,kep:7,kept:18,kind:[7,17,18],kinet:18,kirsch:9,know:[1,2,3,4,5,6],ktran:18,label:[4,7,8,9,10,11,12,13,14,15,16,17,18,19],laplacian:9,larg:12,last:[1,6],later:[0,8,12,18],learn:30,least:[1,2,3,4,5,6,12],lemaitr:31,length:[1,2,3,5,6,18],level:[12,18],lewin:18,libfftw3:30,like:5,list:[1,2,3,4,5,6,15,18,27],load:[7,8,9,10,11,12,13,14,15,16,17,18,19],load_from_pickl:[7,8,9,10,11,12,13,14,15,16,17,18,19],locat:[1,2,3,4,5,6],macdonald:18,magnet:18,magnitud:9,mahota:30,major:18,make:[18,30],manifesto:29,map:18,matrix:[7,8,9,10,11,12,13,14,15,16,17,18,19],matthia:18,max:18,max_:[1,5,6],max_series_:[2,3],max_series_list_:[2,3],maximum:[1,2,3,5,6,12,18],mean:18,measur:18,median:18,medic:0,medicin:18,metadata:2,metadata_:[1,2,6],might:12,min_:[1,5,6],min_series_:[2,3],min_series_list_:[2,3],minimum:[1,2,3,5,6,18],mixtur:18,mmol:18,modal:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25],model:[7,18],modifi:7,moment:18,monogen:15,more:18,most:18,mri:[1,2,6,18],multisequencemod:4,n_alpha:11,n_cluster:18,n_featur:[7,8,9,10,11,12,13,14,15,16,17,18,19],n_frequenc:11,n_gamma:11,n_sampl:[7,8,9,10,11,12,13,14,15,16,17,18,19],n_seri:[1,2,3,5,6,18],n_serie_:[2,3,4],name:9,nb_bin:[1,2,3,5,6],ndarrai:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,28],nearest:28,need:[1,2,3,5,6,7,10,13,14,16,18,19,21,30],nn_valu:28,non_zero_sampl:24,none:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23],nonlinear:18,normal:[1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],normalizatio:[7,8,9,10,11,12,13,14,15,16,17,18,19],note:[1,4,5,6,18],npy:26,number:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],numpi:18,object:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25],obtain:18,oct:5,onc:[1,2,3,4,5,6],onli:[4,18],open:29,optim:18,option:[1,2,3,4,5,6,7,8,9,12,15,17,18],order:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],org:30,organis:[1,6],orient:9,origin:[1,6,7],otherwis:[1,2,3,5,6],out:16,output:[4,12],output_typ:4,over:[2,3,18],overidden:2,overrid:[2,3,4,5],overwrit:4,page:29,paramet:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],parent:25,parent_mod:25,parent_modality_info:25,parker:18,parti:30,partial:18,patch:[8,12],patch_siz:[8,12],path:[1,2,3,4,5,6,27],path_data:[1,2,3,4,5,6,27],path_data_:[1,2,3,4,5,6],pdf:[1,2,3,5,6],pdf_:[1,5,6],pdf_data:[1,6],pdf_list:2,pdf_series_:[2,3],percentag:18,perfect:29,perfectli:18,perfus:18,permiss:18,person:31,phase:15,phasepack:[15,30],physic:18,pickl:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],plai:[1,5,6],plane:11,plasma:18,pleas:30,plot:[1,2,3,5,6],plu:18,point:[17,18],popul:18,population_based_aif:18,posit:[4,7,8,9,10,11,12,13,14,15,16,17,18,19],possibl:[1,2,3,5,6],potenti:18,precis:18,prefer:12,prewitt:9,problem:[18,31],prone:29,properli:2,propos:[7,18],provid:15,pull:30,puls:18,pyfftw:30,pyksvd:30,python:30,quantit:18,queri:31,rais:[25,31],random:18,random_st:[7,14,16,18,19],randomst:18,rang:18,rather:12,ratio:18,raw:5,read:[1,2,3,4,5,6],read_data_from_path:[1,2,3,4,5,6],recurs:30,redifin:[1,5,6],refer:[4,15,17,18],region:18,regular:[15,18],rel:16,relax:18,repetit:18,repres:[1,2,3,6],request:30,requir:12,resolut:18,reson:18,robert:18,roi:[1,2,6],roi_data:[1,2,6,24],roi_data_:[7,8,9,10,11,12,13,14,15,16,17,18,19],rotat:11,s_pre_contrast:18,same:23,sampl:[7,8,9,10,11,12,13,14,15,16,17,18,19],save:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],save_to_pickl:[7,8,9,10,11,12,13,14,15,16,17,18,19],scale:18,scale_sigma:11,schabel:18,scharr:9,scienc:29,scikit:30,search:[28,29],second:2,section:31,seed:18,segment:18,select:18,self:[1,2,3,4,5,6],sequenc:[1,2,3,4,6,18],sequuenc:3,seri:[1,2,3,5,6],setup:30,shape:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24],shift:2,should:[7,8,9,10,11,12,13,14,15,16,17,18,19,29],sigma:18,sigmoid:18,signal:[7,9,10,11,13,14,15,16,18,19],signal_to_conc:18,simpleitk:30,sinc:[1,4,5,6],singl:[1,6,9,17],size:[8,12],slice:17,slide:[8,12],sobel:9,sourc:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],spatial:17,specifi:[2,5],speed:15,spoil:18,standalonemod:[8,9,11,12,13,15,17],standard:11,start:[18,29],start_enh_:18,statist:[1,2,3,4,5,6],store:[1,3,4,5,6],str:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,27],straightforward:30,string:[1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24],studdi:18,sum:29,support:29,sure:30,system:17,sz_data:5,t10:18,t10_:18,t2w:[1,6],taken:18,target:28,tau:[16,18],templat:23,template_mod:23,temporalmod:[7,8,9,10,11,12,13,14,15,16,17,18,19],test:29,than:[12,18,23],therefor:9,thi:[0,2,3,4,12,18,30],third:[18,30],thres_sel:18,threshold:18,through:30,time:[1,2,3,6,18],time_info_:2,toft:18,toolbox:0,tr_:18,transform:[7,8,9,10,11,12,13,14,15,16,17,18,19],tricki:[1,5,6],truth:[4,7,8,9,10,11,12,13,14,15,16,17,18,19,24],tupl:[1,2,5,6,8,12,18],type:[4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,24],type_filt:15,typic:12,uint8:5,unbalanceddataset:30,uncertainti:18,unit:[18,30],updat:[1,6],update:[1,2,3,5,6],update_histogram:[1,2,3,4,5,6],user:29,valu:[5,9,12,18,28],variou:0,vector:11,vein:18,version:30,volum:[1,2,3,4,5,6,18],volume_s:9,voxel:18,want:30,wash:16,watson:18,well:4,when:[2,18],where:[2,5,12],which:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,28],whole:[7,8,9,10,11,12,13,14,15,16,17,18,19],width:18,wish:30,without:18,you:[30,31],your:[30,31],zone:18},titles:["<cite>protoclass</cite> API","protoclass.data_management.ADCModality","protoclass.data_management.DCEModality","protoclass.data_management.DWIModality","protoclass.data_management.GTModality","protoclass.data_management.OCTModality","protoclass.data_management.T2WModality","protoclass.extraction.BrixQuantificationExtraction","protoclass.extraction.DCTExtraction","protoclass.extraction.EdgeSignalExtraction","protoclass.extraction.EnhancementSignalExtraction","protoclass.extraction.GaborBankExtraction","protoclass.extraction.HaralickExtraction","protoclass.extraction.IntensitySignalExtraction","protoclass.extraction.PUNQuantificationExtraction","protoclass.extraction.PhaseCongruencyExtraction","protoclass.extraction.SemiQuantificationExtraction","protoclass.extraction.SpatialExtraction","protoclass.extraction.ToftsQuantificationExtraction","protoclass.extraction.WeibullQuantificationExtraction","protoclass.utils.check_filename_pickle_load","protoclass.utils.check_filename_pickle_save","protoclass.utils.check_img_filename","protoclass.utils.check_modality","protoclass.utils.check_modality_gt","protoclass.utils.check_modality_inherit","protoclass.utils.check_npy_filename","protoclass.utils.check_path_data","protoclass.utils.find_nearest","Welcome to protoclass&#8217;s documentation!","Getting Started","Support"],titleterms:{"class":0,"function":0,adcmodal:1,api:0,brixquantificationextract:7,check_filename_pickle_load:20,check_filename_pickle_sav:21,check_img_filenam:22,check_mod:23,check_modality_gt:24,check_modality_inherit:25,check_npy_filenam:26,check_path_data:27,contact:31,contribut:30,coverag:30,data:0,data_manag:[1,2,3,4,5,6],dcemodal:2,dctextraction:8,document:29,dwimodal:3,edgesignalextract:9,enhancementsignalextract:10,extract:[7,8,9,10,11,12,13,14,15,16,17,18,19],extraction:0,find_nearest:28,gaborbankextract:11,get:30,gtmodal:4,haralickextract:12,indice:29,install:30,intensitysignalextract:13,manag:0,method:0,modul:0,mrsi:0,octmodal:5,phasecongruencyextract:15,pre:0,process:0,protoclass:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],punquantificationextract:14,relat:0,semiquantificationextract:16,spatialextract:17,standalon:0,start:30,support:31,t2wmodal:6,tabl:29,tempor:0,test:30,toftsquantificationextract:18,util:[0,20,21,22,23,24,25,26,27,28],weibullquantificationextract:19,welcom:29}})