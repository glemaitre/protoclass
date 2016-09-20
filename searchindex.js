Search.setIndex({envversion:49,filenames:["api","generated/protoclass.data_management.ADCModality","generated/protoclass.data_management.DCEModality","generated/protoclass.data_management.DWIModality","generated/protoclass.data_management.GTModality","generated/protoclass.data_management.OCTModality","generated/protoclass.data_management.T2WModality","generated/protoclass.extraction.BrixQuantificationExtraction","generated/protoclass.extraction.DCTExtraction","generated/protoclass.extraction.EdgeSignalExtraction","generated/protoclass.extraction.EnhancementSignalExtraction","generated/protoclass.extraction.GaborBankExtraction","generated/protoclass.extraction.HaralickExtraction","generated/protoclass.extraction.IntensitySignalExtraction","generated/protoclass.extraction.PUNQuantificationExtraction","generated/protoclass.extraction.PhaseCongruencyExtraction","generated/protoclass.extraction.SemiQuantificationExtraction","generated/protoclass.extraction.SpatialExtraction","generated/protoclass.extraction.ToftsQuantificationExtraction","generated/protoclass.extraction.WeibullQuantificationExtraction","generated/protoclass.preprocessing.GaussianNormalization","generated/protoclass.preprocessing.PiecewiseLinearNormalization","generated/protoclass.preprocessing.RicianNormalization","generated/protoclass.preprocessing.StandardTimeNormalization","generated/protoclass.utils.check_filename_pickle_load","generated/protoclass.utils.check_filename_pickle_save","generated/protoclass.utils.check_img_filename","generated/protoclass.utils.check_modality","generated/protoclass.utils.check_modality_gt","generated/protoclass.utils.check_modality_inherit","generated/protoclass.utils.check_npy_filename","generated/protoclass.utils.check_path_data","generated/protoclass.utils.find_nearest","index","install","support"],objects:{"protoclass.data_management":{ADCModality:[1,1,1,""],DCEModality:[2,1,1,""],DWIModality:[3,1,1,""],GTModality:[4,1,1,""],OCTModality:[5,1,1,""],T2WModality:[6,1,1,""]},"protoclass.data_management.ADCModality":{__init__:[1,2,1,""],get_pdf:[1,2,1,""],is_read:[1,2,1,""],read_data_from_path:[1,2,1,""],update_histogram:[1,2,1,""]},"protoclass.data_management.DCEModality":{__init__:[2,2,1,""],build_heatmap:[2,2,1,""],get_pdf_list:[2,2,1,""],is_read:[2,2,1,""],read_data_from_path:[2,2,1,""],update_histogram:[2,2,1,""]},"protoclass.data_management.DWIModality":{__init__:[3,2,1,""],is_read:[3,2,1,""],read_data_from_path:[3,2,1,""],update_histogram:[3,2,1,""]},"protoclass.data_management.GTModality":{__init__:[4,2,1,""],extract_gt_data:[4,2,1,""],is_read:[4,2,1,""],read_data_from_path:[4,2,1,""],update_histogram:[4,2,1,""]},"protoclass.data_management.OCTModality":{__init__:[5,2,1,""],is_read:[5,2,1,""],read_data_from_path:[5,2,1,""],update_histogram:[5,2,1,""]},"protoclass.data_management.T2WModality":{__init__:[6,2,1,""],get_pdf:[6,2,1,""],is_read:[6,2,1,""],read_data_from_path:[6,2,1,""],update_histogram:[6,2,1,""]},"protoclass.extraction":{BrixQuantificationExtraction:[7,1,1,""],DCTExtraction:[8,1,1,""],EdgeSignalExtraction:[9,1,1,""],EnhancementSignalExtraction:[10,1,1,""],GaborBankExtraction:[11,1,1,""],HaralickExtraction:[12,1,1,""],IntensitySignalExtraction:[13,1,1,""],PUNQuantificationExtraction:[14,1,1,""],PhaseCongruencyExtraction:[15,1,1,""],SemiQuantificationExtraction:[16,1,1,""],SpatialExtraction:[17,1,1,""],ToftsQuantificationExtraction:[18,1,1,""],WeibullQuantificationExtraction:[19,1,1,""]},"protoclass.extraction.BrixQuantificationExtraction":{__init__:[7,2,1,""],fit:[7,2,1,""],load_from_pickles:[7,2,1,""],save_to_pickles:[7,2,1,""],transform:[7,2,1,""]},"protoclass.extraction.DCTExtraction":{__init__:[8,2,1,""],fit:[8,2,1,""],load_from_pickles:[8,2,1,""],save_to_pickles:[8,2,1,""],transform:[8,2,1,""]},"protoclass.extraction.EdgeSignalExtraction":{__init__:[9,2,1,""],fit:[9,2,1,""],load_from_pickles:[9,2,1,""],save_to_pickles:[9,2,1,""],transform:[9,2,1,""]},"protoclass.extraction.EnhancementSignalExtraction":{__init__:[10,2,1,""],fit:[10,2,1,""],load_from_pickles:[10,2,1,""],save_to_pickles:[10,2,1,""],transform:[10,2,1,""]},"protoclass.extraction.GaborBankExtraction":{__init__:[11,2,1,""],fit:[11,2,1,""],load_from_pickles:[11,2,1,""],save_to_pickles:[11,2,1,""],transform:[11,2,1,""]},"protoclass.extraction.HaralickExtraction":{__init__:[12,2,1,""],fit:[12,2,1,""],load_from_pickles:[12,2,1,""],save_to_pickles:[12,2,1,""],transform:[12,2,1,""]},"protoclass.extraction.IntensitySignalExtraction":{__init__:[13,2,1,""],fit:[13,2,1,""],load_from_pickles:[13,2,1,""],save_to_pickles:[13,2,1,""],transform:[13,2,1,""]},"protoclass.extraction.PUNQuantificationExtraction":{__init__:[14,2,1,""],fit:[14,2,1,""],load_from_pickles:[14,2,1,""],save_to_pickles:[14,2,1,""],transform:[14,2,1,""]},"protoclass.extraction.PhaseCongruencyExtraction":{__init__:[15,2,1,""],fit:[15,2,1,""],load_from_pickles:[15,2,1,""],save_to_pickles:[15,2,1,""],transform:[15,2,1,""]},"protoclass.extraction.SemiQuantificationExtraction":{__init__:[16,2,1,""],fit:[16,2,1,""],load_from_pickles:[16,2,1,""],save_to_pickles:[16,2,1,""],transform:[16,2,1,""]},"protoclass.extraction.SpatialExtraction":{__init__:[17,2,1,""],fit:[17,2,1,""],load_from_pickles:[17,2,1,""],save_to_pickles:[17,2,1,""],transform:[17,2,1,""]},"protoclass.extraction.ToftsQuantificationExtraction":{__init__:[18,2,1,""],compute_aif:[18,3,1,""],conc_to_signal:[18,2,1,""],fit:[18,2,1,""],load_from_pickles:[18,2,1,""],population_based_aif:[18,3,1,""],save_to_pickles:[18,2,1,""],signal_to_conc:[18,2,1,""],transform:[18,2,1,""]},"protoclass.extraction.WeibullQuantificationExtraction":{__init__:[19,2,1,""],fit:[19,2,1,""],load_from_pickles:[19,2,1,""],save_to_pickles:[19,2,1,""],transform:[19,2,1,""]},"protoclass.preprocessing":{GaussianNormalization:[20,1,1,""],PiecewiseLinearNormalization:[21,1,1,""],RicianNormalization:[22,1,1,""],StandardTimeNormalization:[23,1,1,""]},"protoclass.preprocessing.GaussianNormalization":{__init__:[20,2,1,""],denormalize:[20,2,1,""],fit:[20,2,1,""],load_from_pickles:[20,2,1,""],normalize:[20,2,1,""],save_to_pickles:[20,2,1,""]},"protoclass.preprocessing.PiecewiseLinearNormalization":{__init__:[21,2,1,""],denormalize:[21,2,1,""],fit:[21,2,1,""],load_from_pickles:[21,2,1,""],load_model:[21,2,1,""],normalize:[21,2,1,""],partial_fit_model:[21,2,1,""],save_model:[21,2,1,""],save_to_pickles:[21,2,1,""]},"protoclass.preprocessing.RicianNormalization":{__init__:[22,2,1,""],denormalize:[22,2,1,""],fit:[22,2,1,""],load_from_pickles:[22,2,1,""],normalize:[22,2,1,""],save_to_pickles:[22,2,1,""]},"protoclass.preprocessing.StandardTimeNormalization":{__init__:[23,2,1,""],denormalize:[23,2,1,""],fit:[23,2,1,""],load_from_pickles:[23,2,1,""],load_model:[23,2,1,""],normalize:[23,2,1,""],partial_fit_model:[23,2,1,""],save_model:[23,2,1,""],save_to_pickles:[23,2,1,""]},"protoclass.utils":{check_filename_pickle_load:[24,4,1,""],check_filename_pickle_save:[25,4,1,""],check_img_filename:[26,4,1,""],check_modality:[27,4,1,""],check_modality_gt:[28,4,1,""],check_modality_inherit:[29,4,1,""],check_npy_filename:[30,4,1,""],check_path_data:[31,4,1,""],find_nearest:[32,4,1,""]},protoclass:{data_management:[0,0,0,"-"],extraction:[0,0,0,"-"],preprocessing:[0,0,0,"-"],utils:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","staticmethod","Python static method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:staticmethod","4":"py:function"},terms:{"1st":18,"2nd":18,"boolean":[20,21,22],"case":18,"default":[1,2,3,4,5,6,7,8,9,12,15,17,18,20,21,22,23],"final":[18,34],"float":[1,2,3,5,6,18,20,22,23],"int":[1,2,3,4,5,6,8,12,18,21,23,32],"return":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],"static":18,"true":[1,2,3,4,5,6,18,21,23],"while":[1,2,6],__init__:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],accord:17,acquisit:2,adc:1,address:[18,35],addtional:0,affect:[4,23],agent:18,aif:18,aif_param:18,align:[21,23],all:[2,3,17],allow:0,almost:34,alpha:[11,18,23],alreadi:[20,21,22],also:2,amplitud:18,anaconda:34,angl:[11,18],ani:35,anoth:29,aorta:18,appear:18,appli:[0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],approach:18,apt:34,area:18,argument:[4,12],arrai:[2,5,9,21,32],arteri:18,associ:[1,2,4,6,18,32],attribut:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],auc:16,auto:[1,2,3,6,20,22],avail:[0,9],averag:18,axi:18,azimuth:9,b_seq:3,bank:11,base:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],base_mod:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],base_modality_:[8,9,10,11,12,13,15,17,20,22,23],becaus:33,becom:18,beeen:23,been:[1,2,3,4,5,6,18,20,21,22,23,25],befor:[18,34],beta:[14,18],between:18,bia:18,bin:[1,2,3,5,6,12],bin_:[1,5,6],bin_data:[1,6],bin_list:2,bin_series_:[2,3],bins_heatmap:2,biologi:18,bit:[5,12],blood:18,bool:[1,2,3,4,5,6,18,20,21,22,23],bpp:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],brix:7,bucklei:18,build_heatmap:2,buonaccorsi:18,c_t:18,call:[1,6,20,21,22],can:[1,2,4,5,6,9,15,17,18,34,35],cannot:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],cat:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],cat_gt:4,cat_gt_:4,categor:4,cb_t:18,cb_t_:18,center:18,centr:17,check:[24,25,26,27,28,29,30,31,35],cheung:18,child:29,child_mod:29,child_object:29,choic:4,circl:18,classif:0,classinfo:29,clone:34,cluster:18,code:[5,34],cogito:33,com:34,come:34,comput:[1,2,3,4,5,6,8,9,11,12,15,17,18],compute_aif:18,conc:18,conc_to_sign:18,concentr:18,conda:34,congruenc:15,consid:[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],consist:28,constant:18,constructor:[2,3,4,5],contact:33,contain:[1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19],contan:15,content:33,continu:34,contour:17,contrast:18,contribut:33,cooccur:[8,12],coord_system:17,coordin:17,correspond:[4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,28],could:7,count:12,coverag:33,cp_t_:18,current:21,cylindr:17,cython:34,dale:18,data_:[1,2,3,4,5,6,9,17],data_manag:0,davi:18,dce:[2,3,18],dce_mod:18,dct:8,decai:18,deg:18,delai:18,denni:18,denorm:[20,21,22,23],depend:[4,34],deriv:18,desir:[4,17],detect:18,determin:18,dev:34,deviat:[11,20,22,23],diamet:18,dict:[1,2,6,15,20,22,23],dict_param:15,dictionari:[2,15,20,22,23],dictionnari:[1,6],differ:[1,2,3,4,5,6,11,15,34],dimens:5,direct:11,distanc:[12,17,18],divid:21,document:0,done:18,drive:23,dtype:[5,32],duerk:18,dure:[0,20,21,22,23],dwi:3,dynam:18,each:[1,2,3,4,5,6,11,18],eccentr:18,echo:18,edge:[9,11,15],edge_detector:9,effect:18,eigen3:34,either:[15,17,18,21,23],element:[1,2,6],elev:9,ellips:18,enhanc:[16,18],enhancement:[7,10,14,16,18,19],ensur:34,equal:[7,8,9,10,11,12,13,14,15,16,17,18,19],equival:[2,23],ergo:33,error:[29,33],estim:[18,20,22,23],euclidean:17,exist:31,exp:23,experiment:18,exponenti:[18,23],extend:18,extens:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25],extract:[0,1,2,4,6],extract_gt_data:4,factor:23,fals:[18,21,23],featur:[0,7,8,9,10,11,12,13,14,15,16,17,18,19],file:[5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,30],filenam:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,30],filer:11,filter:[9,11,15,23],find:[7,10,13,14,16,18,19,20,21,22,23,32],first:[1,2,3,5,6],fit:[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],fit_aif:18,fit_params_:[20,21,22,23],fittinf:23,fix:18,flash:18,flexibl:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],flip:18,flip_angl:18,flow:33,focal:18,folder:[1,3,4,6],follow:[0,7,14,16,18,19,20,22,23,35],form:18,format:0,found:[1,2,3,5,6,18,23,32],frequenc:11,from:[1,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,29,34],full:[0,4,34],gabor:11,gabor_filter_3d:0,gamma:11,gaussian:[18,20,23],gener:[5,18],get:33,get_pdf:[1,6],get_pdf_list:2,git:34,github:[34,35],given:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23],glemaitr:34,global:18,gradient:[9,18],graph:23,greater:18,grei:12,ground:[4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],ground_truth:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],guassian:18,guillaum:35,handl:[0,1,2,3,4,5,6],haralick:12,have:[1,2,3,4,5,6,18,23],heatmap:2,hematocrit:18,hematocrit_:18,hematrocrit:18,high:18,higher:12,hillenbrand:18,histogram:[1,2,3,4,5,6],hoffmann:7,horizont:23,http:34,human:33,idx:32,iii:9,imag:[0,1,2,3,4,5,6,8,9,11,12,15,17,34],image:18,img:[5,26],includ:[0,18],index:[4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,32,33],indic:12,indice:[1,2,6],infer:[18,23],info:29,inform:[1,2,4,6,17,21,23],inher:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],inherit:29,initi:[18,20,22,23],inject:18,input:[12,18],insid:[2,21,23,32],instal:34,install:33,insterest:28,integ:[1,2,3,5,6,12,18],integr:34,intens:[1,2,3,5,6,23],intensiti:13,intensity_rang:2,interest:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27,28],interv:18,is_fitted_:[20,21,22,23],is_model_fitted_:23,is_read:[1,2,3,4,5,6],issu:35,jackson:18,jayson:18,jesberg:18,journal:18,kei:[20,22,23],kel:7,kep:7,kept:18,kind:[7,17,18],kinet:18,kirsch:9,know:[1,2,3,4,5,6,20,21,22],ktran:18,label:[4,7,8,9,10,11,12,13,14,15,16,17,18,19],landmark:21,landmarks_model:21,laplacian:9,larg:12,last:[1,6],later:[0,8,12,18],learn:34,least:[1,2,3,4,5,6,12],lemaitr:35,length:[1,2,3,5,6,18],level:[12,18],lewin:18,libfftw3:34,like:5,linear:21,list:[1,2,3,4,5,6,15,18,31],load:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],load_from_pickl:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],load_model:[21,23],locat:[1,2,3,4,5,6],macdonald:18,magnet:18,magnitud:9,mahota:34,major:18,make:[18,34],manifesto:33,map:18,match:21,matrix:[7,8,9,10,11,12,13,14,15,16,17,18,19],matthia:18,max:18,max_:[1,5,6],max_it:23,max_series_:[2,3],max_series_list_:[2,3],maximum:[1,2,3,5,6,12,18,23],mean:[18,20,22],measur:18,median:18,medic:0,medicin:18,metadata:2,metadata_:[1,2,6],might:12,min_:[1,5,6],min_series_:[2,3],min_series_list_:[2,3],minimum:[1,2,3,5,6,18],mixtur:18,mmol:18,modal:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27,28,29],model:[7,18,21,23],modifi:7,moment:18,monogen:15,more:18,most:18,mri:[1,2,6,18],multisequencemod:4,n_alpha:11,n_cluster:18,n_featur:[7,8,9,10,11,12,13,14,15,16,17,18,19],n_frequenc:11,n_gamma:11,n_sampl:[7,8,9,10,11,12,13,14,15,16,17,18,19],n_seri:[1,2,3,5,6,18],n_serie_:[2,3,4],name:9,nb_bin:[1,2,3,5,6],nb_landmark:21,ndarrai:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,32],nearest:32,need:[1,2,3,5,6,7,10,13,14,16,18,19,20,21,22,23,25,34],nn_valu:32,non_zero_sampl:28,none:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27],nonlinear:18,normal:[1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],normalizatio:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],note:[1,4,5,6,18],npy:[21,23,30],number:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23],numpi:18,object:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27,28,29],obtain:18,oct:5,off:22,offset:22,onc:[1,2,3,4,5,6],onli:[4,18],online:[21,23],open:33,optim:18,option:[1,2,3,4,5,6,7,8,9,12,15,17,18,20,21,22,23],order:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],org:34,organis:[1,6],orient:9,origin:[1,6,7],otherwis:[1,2,3,5,6],out:16,output:[4,12],output_typ:4,over:[2,3,18],overidden:2,overrid:[2,3,4,5],overwrit:4,page:33,param:[20,22,23],paramet:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],parent:29,parent_mod:29,parent_modality_info:29,parker:18,parti:34,partial:18,partial_fit_model:[21,23],patch:[8,12],patch_siz:[8,12],path:[1,2,3,4,5,6,21,23,31],path_data:[1,2,3,4,5,6,31],path_data_:[1,2,3,4,5,6],pdf:[1,2,3,5,6],pdf_:[1,5,6],pdf_data:[1,6],pdf_list:2,pdf_series_:[2,3],penal:23,percentag:18,perfect:33,perfectli:18,perfus:18,permiss:18,person:35,phase:15,phasepack:[15,34],physic:18,pickl:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],piecewis:21,plai:[1,5,6],plane:11,plasma:18,pleas:34,plot:[1,2,3,5,6],plu:18,point:[17,18],popul:18,population_based_aif:18,posit:[4,7,8,9,10,11,12,13,14,15,16,17,18,19],possibl:[1,2,3,5,6],potenti:18,precis:18,prefer:12,preprocess:0,prewitt:9,problem:[18,35],prone:33,properli:2,propos:[7,18],provid:15,pull:34,puls:18,pyfftw:34,pyksvd:34,python:34,quantit:18,queri:35,rais:[29,35],random:18,random_st:[7,14,16,18,19],randomst:18,rang:18,rather:12,ratio:18,raw:5,read:[1,2,3,4,5,6],read_data_from_path:[1,2,3,4,5,6],recurs:34,redifin:[1,5,6],refer:[4,15,17,18],refit:[21,23],region:18,regular:[15,18],rel:16,relax:18,repetit:18,repres:[1,2,3,6],request:34,requir:12,resolut:18,reson:18,rician:22,rmse:[21,23],robert:18,roi:[1,2,6],roi_data:[1,2,6,28],roi_data_:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],rotat:11,s_pre_contrast:18,same:27,sampl:[7,8,9,10,11,12,13,14,15,16,17,18,19],save:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],save_model:[21,23],save_to_pickl:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],scale:[18,23],scale_sigma:11,schabel:18,scharr:9,scienc:33,scikit:34,search:[32,33],second:2,section:35,seed:18,segment:18,select:18,self:[1,2,3,4,5,6,20,21,22,23],sequenc:[1,2,3,4,6,18],sequuenc:3,seri:[1,2,3,5,6,23],setup:34,shape:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],shift:[2,23],should:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,33],show:[21,23],sigma:[18,20,22],sigmoid:18,signal:[7,9,10,11,13,14,15,16,18,19,21],signal_to_conc:18,simpleitk:34,sinc:[1,4,5,6],singl:[1,6,9,17],size:[8,12],slice:17,slide:[8,12],sobel:9,some:0,sourc:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],spatial:17,specifi:[2,5,20,22,23],speed:15,spoil:18,stanalonemod:20,standalinemod:21,standalonemod:[8,9,11,12,13,15,17,20,21,22,23],standard:[11,20,22,23],start:[18,33],start_enh_:18,statist:[1,2,3,4,5,6],std:23,store:[1,3,4,5,6,21,23],str:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31],straightforward:34,string:[1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],studdi:18,sum:33,support:33,sure:34,system:17,sz_data:5,t10:18,t10_:18,t2w:[1,6],taken:18,target:32,tau:[16,18],templat:[23,27],template_mod:27,temporalmod:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23],test:33,than:[12,18,27],thei:[20,22,23],therefor:9,thi:[0,2,3,4,12,18,34],third:[18,34],thres_sel:18,threshold:18,through:[23,34],time:[1,2,3,6,18,21,23],time_info_:2,toft:18,toolbox:0,tr_:18,transform:[7,8,9,10,11,12,13,14,15,16,17,18,19],tricki:[1,5,6],truth:[4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],tupl:[1,2,5,6,8,12,18],type:[4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28],type_filt:15,typic:12,uint8:5,unbalanceddataset:34,uncertainti:18,unit:[18,34],updat:[1,6,21],update:[1,2,3,5,6],update_histogram:[1,2,3,4,5,6],user:33,valu:[5,9,12,18,20,22,23,32],variou:0,vector:11,vein:18,verbos:[21,23],version:34,vertic:23,volum:[1,2,3,4,5,6,18],volume_s:9,voxel:18,walk:23,want:34,wash:16,watson:18,weight:23,well:4,when:[2,18,23],where:[2,5,12,21,23],whether:[21,23],which:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,32],whole:[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],width:18,wish:34,without:18,you:[34,35],your:[34,35],zone:18},titles:["<cite>protoclass</cite> API","protoclass.data_management.ADCModality","protoclass.data_management.DCEModality","protoclass.data_management.DWIModality","protoclass.data_management.GTModality","protoclass.data_management.OCTModality","protoclass.data_management.T2WModality","protoclass.extraction.BrixQuantificationExtraction","protoclass.extraction.DCTExtraction","protoclass.extraction.EdgeSignalExtraction","protoclass.extraction.EnhancementSignalExtraction","protoclass.extraction.GaborBankExtraction","protoclass.extraction.HaralickExtraction","protoclass.extraction.IntensitySignalExtraction","protoclass.extraction.PUNQuantificationExtraction","protoclass.extraction.PhaseCongruencyExtraction","protoclass.extraction.SemiQuantificationExtraction","protoclass.extraction.SpatialExtraction","protoclass.extraction.ToftsQuantificationExtraction","protoclass.extraction.WeibullQuantificationExtraction","protoclass.preprocessing.GaussianNormalization","protoclass.preprocessing.PiecewiseLinearNormalization","protoclass.preprocessing.RicianNormalization","protoclass.preprocessing.StandardTimeNormalization","protoclass.utils.check_filename_pickle_load","protoclass.utils.check_filename_pickle_save","protoclass.utils.check_img_filename","protoclass.utils.check_modality","protoclass.utils.check_modality_gt","protoclass.utils.check_modality_inherit","protoclass.utils.check_npy_filename","protoclass.utils.check_path_data","protoclass.utils.find_nearest","Welcome to protoclass&#8217;s documentation!","Getting Started","Support"],titleterms:{"class":0,"function":0,adcmodal:1,api:0,brixquantificationextract:7,check_filename_pickle_load:24,check_filename_pickle_sav:25,check_img_filenam:26,check_mod:27,check_modality_gt:28,check_modality_inherit:29,check_npy_filenam:30,check_path_data:31,contact:35,contribut:34,coverag:34,data:0,data_manag:[1,2,3,4,5,6],dcemodal:2,dctextraction:8,document:33,dwimodal:3,edgesignalextract:9,enhancementsignalextract:10,extract:[7,8,9,10,11,12,13,14,15,16,17,18,19],extraction:0,find_nearest:32,gaborbankextract:11,gaussiannorm:20,get:34,gtmodal:4,haralickextract:12,indice:33,install:34,intensitysignalextract:13,manag:0,method:0,modul:0,octmodal:5,phasecongruencyextract:15,piecewiselinearnorm:21,pre:0,preprocess:[20,21,22,23],process:0,protoclass:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],punquantificationextract:14,relat:0,riciannorm:22,semiquantificationextract:16,spatialextract:17,standalon:0,standardtimenorm:23,start:34,support:35,t2wmodal:6,tabl:33,tempor:0,test:34,toftsquantificationextract:18,util:[0,24,25,26,27,28,29,30,31,32],weibullquantificationextract:19,welcom:33}})