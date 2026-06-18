| Segment | Any          | Anomaly     | without        |               | Training   | via |
| ------- | ------------ | ----------- | -------------- | ------------- | ---------- | --- |
|         | Hybrid       | Prompt      | Regularization |               |            |     |
|         | YunkangCao1∗ | XiaohaoXu1∗ |                | ChenSun1      | YuqiCheng1 |     |
|         | ZongweiDu1   | LiangGao1   |                | WeimingShen1§ |            |     |
3202 yaM 81  ]VC.sc[  1v42701.5032:viXra 1StateKeyLaboratoryofDigitalManufacturingEquipmentandTechnology,
HuazhongUniversityofScienceandTechnology,China
| {cyk_hust, | sun_chen, | chengyuqi, | duzongwei, |     | gaoliang}@hust.edu.cn |     |
| ---------- | --------- | ---------- | ---------- | --- | --------------------- | --- |
xxh11102019@outlook.com,chengyuqi.c@qq.com,wshen@ieee.org
Abstract
Wepresentanovelframework,i.e.,SegmentAnyAnomaly+(SAA+),forzero-
| shot                                  | anomaly segmentation | with hybrid | prompt                            | regularization | to improve | the |
| ------------------------------------- | -------------------- | ----------- | --------------------------------- | -------------- | ---------- | --- |
| adaptabilityofmodernfoundationmodels. |                      |             | Existinganomalysegmentationmodels |                |            |     |
typicallyrelyondomain-specificfine-tuning,limitingtheirgeneralizationacross
| countlessanomalypatterns. |     | Inthiswork,inspiredbythegreatzero-shotgeneral- |     |     |     |     |
| ------------------------- | --- | ---------------------------------------------- | --- | --- | --- | --- |
izationabilityoffoundationmodelslikeSegmentAnything,wefirstexploretheir
assemblytoleveragediversemulti-modalpriorknowledgeforanomalylocaliza-
tion. Fornon-parameterfoundationmodeladaptationtoanomalysegmentation,
wefurtherintroducehybridpromptsderivedfromdomainexpertknowledgeand
| targetimagecontextasregularization. |     |     | OurproposedSAA+modelachievesstate- |     |     |     |
| ----------------------------------- | --- | --- | ---------------------------------- | --- | --- | --- |
of-the-artperformanceonseveralanomalysegmentationbenchmarks,including
| VisA,MVTec-AD,MTD,andKSDD2,inthezero-shotsetting. |     |     |     |     | Wewillrelease |     |
| ------------------------------------------------- | --- | --- | --- | --- | ------------- | --- |
thecodeathttps://github.com/caoyunkang/Segment-Any-Anomaly.
1 Introduction
Anomalysegmentationmodels[1,2,3]haveattractedgreatinterestinvariousdomains,e.g,,industrial
qualitycontrol[4,5]andmedicaldiagnoses[6]. Thekeytoreliableanomalysegmentationisto
discriminatethedistributionofanomalydatafromnormaldata. Specifically,thispaperconsiders
zero-shotanomalysegmentation(ZSAS)onimages,whichisapromisingyetunexploredsetting
whereneithernormalnorabnormalimageisprovidedforthetargetcategoryduringtraining.
Duetothescarcityofabnormalsamplesfortraining,manyworksareworkingtowardsunsupervised
or self-supervised anomaly segmentation, which targets learning a representation of the normal
samples during training. Then, the anomalies can be segmented by calculating the discrepancy
betweenthetestsampleandthelearnednormaldistribution. Inspecific,thesemodels,including
auto-encoder-based reconstruction [7, 8, 9, 10, 11, 12], one-class classification [13, 14, 15], and
memory-based normal distribution [3, 2, 16, 17, 18] methods, typically require training separate
modelsforcertainlimitedcategories.However,inreal-worldscenarios,therearemillionsofindustrial
products,anditisnotcost-effectivetocollectalargetrainingsetforindividualobjects,whichhinders
theirdeploymentincaseswhenefficientdeploymentsarerequired,e.g.,theinitialstageofproduction.
Recently,foundationmodels,e.g.,SAM[19]andCLIP[20],exhibitgreatzero-shotvisualperception
abilitiesbyretrievingpriorknowledgestoredinthesemodelsviaprompting[21,22]. Inthiswork,
wewouldliketoexplorehowtoadaptfoundationmodelstorealizeanomalysegmentationunderthe
*EqualContribution.
§CorrespondingAuthor.
Preprint.Underreview.

Figure1: Towardssegmentinganyanomalywithouttraining,wefirstconstructavanillabaseline
(SAA)bypromptingintoacascadeofanomalyregiongenerator(e.g.,aprompt-guidedobjectdetec-
tionfoundationmodel[23])andanomalyregionrefiner(e.g.,asegmentationfoundationmodel[19])
modulesviaanaiveclass-agnosticlanguageprompt(e.g.,“Anomaly”). However,SAAshowsthe
severefalse-alarmproblem,whichfalselydetectsallthe“wick”ratherthantheground-truthanomaly
region(the“overlong wick”). Thus,wefurtherstrengthentheregularizationwithhybridprompts
intherevampedmodel(SAA+),whichsuccessfullyhelpsidentifytheanomalyregion.
zero-shotsetting. Tothisend,asisshowninFig.1,wefirstconstructavanillabaseline,i.e.,Segment
AnyAnomaly(SAA),bycascadingprompt-guidedobjectdetection[23]andsegmentationfoundation
models[19],whichserveasAnomalyRegionGeneratorandAnomalyRegionRefiner,respectively.
Followingthepracticetounlockfoundationmodelknowledge[24,25],naivelanguageprompts,e.g.,
“defect”or“anomaly”,areutilizedtosegmentdesiredanomaliesforatargetimage. Inspecific,the
languagepromptisusedtoprompttheAnomalyRegionGeneratortogenerateprompt-conditioned
box-levelregionsfordesiredanomalyregions. ThentheseregionsarerefinedintheAnomalyRegion
Refinertoproducefinalpredictions,i.e.,masks,foranomalysegmentation.
However,asisshowninFigure1,vanillafoundationmodelassembly(SAA)tendstocausesignificant
falsealarms,e.g.,SAAwronglyreferstoallwicksasanomalieswhereasonlytheoverlongwick
isarealanomaly,whichweattributetotheambiguitybroughtbynaivelanguageprompts. Firstly,
conventionallanguagepromptsmaybecomeineffectivewhenfacingthedomainshiftbetweenthepre-
trainingdatadistributionoffoundationmodelsanddownstreamdatasetsforanomalysegmentation.
Secondly,thedegreeof“anomaly”foratargetdependsontheobjectcontext,whichishardfornaive
coarse-grainedlanguageprompts,e.g.,“an anomaly region”,toexpressexactly.
Thus,goingbeyondnaivelanguageprompts,weincorporatedomainexpertknowledgeandtarget
imagecontextinourrevampedframework, i.e., SegmentAnyAnomaly+(SAA+), respectively.
Ontheonehand, expertknowledgeprovidesdetaileddescriptionsofanomaliesthatarerelevant
tothetargetinopen-worldscenarios. Weutilizemorespecificdescriptionsasin-contextprompts,
effectivelyaligningtheimagecontentinbothpre-trainedandtargetdatasets. Ontheotherhand,we
utilizethetargetimagecontexttoreliablyidentifyandadaptivelycalibrateanomalysegmentation
predictions[26,27]. Byleveragingtherichcontextualinformationpresentinthetargetimage,we
canaccuratelyassociatetheobjectcontextwiththefinalanomalypredictions.
Technically, apart from naive class-agnostic prompts, we leverage domain expert knowledge to
constructtarget-orientedanomalylanguageprompts,i.e.,class-specificlanguageexpressions.Besides,
aslanguagecannotaccuratelyretrieveregionswithcertainobjectcharacteristics,suchasnumber,
size,andlocation,precisely[28,29],weintroduceobjectpropertypromptsintheformofthresholding
filters. Thesepromptsassistinidentifyingandremovingregioncandidatesthatdonotsatisfydesired
properties. Furthermore,tofullyexploitthetargetimagecontext,wesuggestutilizingimagesaliency
andregionconfidencerankingasprompts,whichmodeltheanomalydegreeofaregionbyconsidering
thesimilarities,e.g.,euclideandistance,betweenitandotherregionswithintheimage. Finally,we
conductthoroughexperimentstoconfirmtheefficacyofourhybridpromptsinadaptingfoundation
modelstozero-shotanomalysegmentation. Specifically,ourfinalmodel(SAA+)attainsnewstate-
of-the-artperformanceonvariousanomalysegmentationdatasetsunderthezero-shotsetting. To
summarize,ourmaincontributionsare:
• We propose the SAA framework for anomaly segmentation, allowing the collaborative
assemblyofdiversefoundationmodelswithouttheneedfortraining.
2

• We introduce hybrid prompts as a regularization technique, leveraging domain expert
knowledgeandtargetimagecontexttoadaptfoundationmodelsforanomalysegmentation.
ThisleadstothedevelopmentofSAA+,anenhancedversionofourframework.
• Ourmethodachievesstate-of-the-artperformanceinzero-shotanomalysegmentationon
several benchmark datasets, including VisA, MVTec-AD, KSDD2, and MTD. Notably,
SAA/SAA+ demonstrates remarkable capability in detecting texture-related anomalies
withoutrequiringanyannotation.
2 Relatedwork
Anomaly Segmentation. Due to the limited availability and high cost of abnormal images in
industrialsettings,muchofthecurrentresearchonanomalysegmentationfocusesonunsupervised
methodsthatrelysolelyonnormalimages. Reconstruction-basedapproaches,suchasthoseproposed
in[7,8,9,10,11,12],scoreanomalieswithtrainanencoder-decodermodeltoreconstructimagesfor
segmentationpurposes. Bycomparingtheinputimagewiththereconstructedversion,thesemethods
canpredictthelocationofanomalies. Featureembedding-basedmethods,ontheotherhand,typically
employteacher-studentarchitecture[30,31,32,33,34,35,36,1],one-classclassificationtechnology
[13,14,15],ormemory-basednormaldistribution[3,2,16]tosegmentanomaliesbyidentifying
differencesinfeaturedistributionbetweennormalandabnormalimages.
Recently,researchershavebeguntoexplorethepotentialofZSAS[37,38,39,40],whicheliminates
theneedforeithernormalorabnormalimagesduringthetrainingprocess.Amongthem,WinClip[25]
pioneersthepotentialoffoundationmodels,e.g.,visual-languagemodels,fortheZSALtask. Unlike
WinClip[25]thatsegmentsanomaliesthroughtext-visualsimilarity,weproposetogenerateproposals
andscoretheiranomalydegree,achievingmuchbettersegmentationperformance.
FoundationModel. Foundationmodelsshowanimpressiveabilitytosolvediversevisiontasksina
zero-shotmanner. Specifically,thesemodelscanlearnastrongrepresentationbytrainingonlarge-
scaledatasets [41]. Whileearlywork[20,42]focusondevelopingrobustimage-wiserecognition
capacity, recent work [43, 44, 45, 46, 47, 23] introduce foundation models or their applications
fordensevisualtasks. Forinstance,GroundingDINO[23]achievesencouragingopen-setobject
detectionabilityusingarbitrarytextsasqueries. Recently,SAM[19]demonstratesapowerfulability
toextracthigh-qualityobjectsegmentationmasksintheopenworld. Impressedbythesuccessof
thesefoundationmodels,wewouldliketoexplorehowtoadapttheseoff-the-shelfmodelstodetect
anomalieswithoutanytrainingonthedownstreamdatasetsforanomalysegmentation.
PromptEngineering. Promptengineeringisawidelyemployedtechniquethatinvolvesadapting
foundation models for downstream tasks. Generally, this approach involves appending a set of
learnabletokenstotheinput. Priorstudieshaveinvestigatedpromptingwithtextinputs[48],vision
inputs[49,50,51],andbothtextandvisualinputs[52,53,54].Despitetheireffectivenessinadapting
foundationmodelstovariousdownstreamtasks,promptingmethodscannotbeemployedinZSAS
becausetheyrequiretrainingdata,whichisnotavailableinZSAS.Incontrast,somemethodsemploy
heuristicprompts[55]thatdonotrequireanytraining,makingthemmorefeasiblefortaskswithout
anydata. Inthispaper,weproposeusinghybridpromptsderivedfromdomainexpertknowledgeand
targetimagecontextforZSAS.
3 SAA:VanillaFoundationModelAssemblyforZSAS
3.1 ProblemDefinition: Zero-shotAnomalySegmentation(ZSAS)
The goal of ZSAS is to perform anomaly segmentation on new objects without requiring any
correspondingobjecttrainingdata. ZSASseekstocreateananomalymapA∈[0,1]h×w×1based
onanemptytrainingset∅,inordertoidentifytheanomalydegreeforindividualpixelsinanimage
I∈Rh×w×3thatincludesnovelobjects. TheZSAStaskhasthepotentialtosignificantlyreducethe
needfortrainingdataandlowerthecostsassociatedwithreal-worldinspectiondeployments.
3

Figure 2: Overview of the proposed Segment Any Anomaly + (SAA+) framework. We adapt
foundationmodelstozero-shotanomalysegmentationviahybridpromptregularization. Inspecific,
apartfromnaiveclass-agnosticlanguageprompts,theregularizationcomesfrombothdomainexpert
knowledge,includingmoredetailedclass-specificlanguageandobjectpropertyprompts,andtarget
imagecontext,includingvisualsaliencyandconfidenceranking-relatedprompts.
3.2 BaselineModelAssembly: SegmentAnyAnomaly(SAA)
ForZSAS,westartbyconstructingavanillafoundationmodelassembly,i.e.,SegmentAnyAnomaly
(SAA),asshowninFig. 1. Inspecific,givenacertainqueryimageforanomalysegmentation,we
firstuselanguagesastheinitialprompttoroughlyretrievecoarseanomalyregionproposalsviaan
AnomalyRegionGeneratorimplementedwithalanguage-drivenvisualgroundingfoundationmodel,
i.e.,GroundingDINO[23]. Afterward,anomalyregionproposalsarerefinedintopixel-wisehigh-
qualitysegmentationmaskswiththeAnomalyRegionRefinerinwhichaprompt-drivensegmentation
foundationmodel,i.e.,SAM[19],isused.
3.2.1 AnomalyRegionGenerator
With recent booming development on language-vision models, some foundation models [24, 23,
46] gradually acquire the ability to retrieve objects in images through language prompts. Given
languagepromptsT thatdescribedesiredregionstobedetected,e.g.,“anomaly”,foundationmodels
can generate desired regions for a query image I. There we base the architecture of the region
detectoronatext-guidedopen-setobjectdetectionarchitectureforvisualgrounding. Specifically,we
takeaGroundingDINO[23]architecturethathasbeenpre-trainedonlarge-scalelanguage-vision
datasets[41]. Suchanetworkfirstextractsthefeaturesofthelanguagepromptandthequeryimage
viatextencoderandvisualencoder,respectively. Thentheroughobjectregionsaregeneratedinthe
formofboundingboxeswithacross-modalitydecoder. Giventhebounding-box-levelregionsetRB,
andtheircorrespondingconfidencescoresetS,themoduleofanomalyregiongenerator(Generator)
canbeformulatedas,
RB,S :=Generator(I,T) (1)
3.2.2 AnomalyRegionRefiner
Togeneratepixel-wiseanomalysegmentationresults,weproposeAnomalyRegionRefinertorefine
thebounding-box-levelanomalyregioncandidatesintoananomalysegmentationmaskset. Tothis
end,weuseasophisticatedfoundationmodelforopen-worldvisualsegmentation,i.e.,SAM[19].
ThismodelmainlyincludesaViT-based[56]backboneandaprompt-conditionedmaskdecoder. In
specific,themodelistrainedonalarge-scaleimagesegmentationdataset[19]withonebillionfine-
grainedmasks,whichenableshigh-qualitymaskgenerationabilitiesunderanopen-setsegmentation
4

setting. Theprompt-conditionedmaskdecoderacceptsvarioustypesofpromptsasinput. Weregard
the bounding box candidates RB as prompts and obtain pixel-wise segmentation masks R. The
moduleoftheAnomalyRegionRefiner(Refiner)canbeformulatedasfollows,
R:=Refiner(I,RB) (2)
Till then, we obtain the set of regions in the form of high-quality segmentation masks R with
correspondingconfidencescoresS. Tosumupwesummarizeframework(SAA)asfollows,
R,S :=SAA(I,T ) (3)
n
whereT isanaiveclass-agnosticlanguageprompt,e.g.,“anomaly”,utilizedinSAA.
n
3.3 AnalysisontheZSASPerformanceofVanillaFoundationModelAssembly
We present some preliminary experiments to evaluate the efficacy of vanilla foundation model
assemblyforZSAS.Despitethesimplicityandintuitivenessofthesolution,weobservealanguage
ambiguityissue. Specifically,certainlanguageprompts,suchas“anomaly”,mayfailtodetectthe
desiredanomalyregions. Forinstance,asdepictedinFig. 1,all“wick”iserroneouslyidentifiedas
ananomalybytheSAAwiththe“anomaly”prompt.
Weattributethislanguageambiguitytothedomaingapbetweenthepretraininglanguage-vision
datasetsandthetargetedZSASdatasets,whichmeansthatsomelanguagepromptsmayhavedifferent
meaningsandbeassociatedwithdifferentimagecontentsindistinctdatasets. Inaddition,thereis
hardlyanyadjectiveexpressionlike“anomaly”inthoselarge-scaledatasets,thusmakingthiskindof
promptdesignpooratunderstandingwhatisananomalyregion. Additionally,theexact“anomaly”
isobject-specificandwouldvaryacrossobjects. Forexample,itdenotesthescratchesonleatheror
thecrackonhazelnut. ThelanguageambiguityissueleadstoseverefalsealarmsinZSASdatasets.
Weproposeintroducinghybridpromptsgeneratedbydomainexpertknowledgeandthetargetimage
contexttoreducelanguageambiguity,therebyachievingbetterZSASperformance.
4 SAA+: FoundationModelAdaptionviaHybridPromptRegularization
ToaddresslanguageambiguityinSAAandimproveitsabilityonZSAS,weproposeanupgraded
version called SAA+ that incorporates hybrid prompts, as Fig. 2. In addition to leveraging the
knowledgegainedfrompre-trainedfoundationmodels,SAA+utilizesbothdomainexpertknowledge
andtargetimagecontexttogeneratemoreaccurateanomalyregionmasks. Weprovidefurtherdetails
onthesehybridpromptsbelow.
4.1 PromptGeneratedfromDomainExpertKnowledge
Followingthetrendofpromptlearning[48,54],weinitializetheprompt,whichunlockstheknowl-
edgeoffoundationmodels,intheformoflanguage. However,thelanguageambiguityissuecaused
bythedomaingapisparticularlyseverewhenusingonlythenaivelanguageprompt“anomaly”. To
addressthisproblem,weleveragedomainexpertknowledgethatcontainsusefulpriorinformation
aboutthetargetanomalyregions. Specifically,althoughexpertsmaynotprovideacomprehensive
listofpotentialopen-worldanomaliesforanewproduct,theycanidentifysomecandidatesbasedon
theirpastexperienceswithsimilarproducts. Domainexpertknowledgeenablesustorefinethenaive
“anomaly”promptintomorespecificpromptsthatdescribetheanomalystateingreaterdetail. In
additiontolanguageprompts,weintroducepropertypromptstocomplementthelackofawareness
onspecificpropertieslike“count”and“area” [28]inexistingfoundationmodels[28].
4.1.1 AnomalyLanguageExpressionasPrompt
Todescribepotentialopen-worldanomalies,weproposedesigningmorepreciselanguageprompts.
Thesepromptsarecategorizedintotwotypes: class-agnosticandclass-specificprompts.
Class-agnosticprompts(T )aregeneralpromptsthatdescribeanomaliesthatarenotspecifictoany
a
particularcategory,e.g.,“anomaly”and“defect”. Despitethedomaingapbetweenthepre-trained
datasetsandthetargetedZSASdatasets,ourempiricalanalysis(5.3)showsthatthesegenericprompts
provideencouraginginitialperformance.
5

Class-specific prompts (T ) are designed based on expert knowledge of abnormal patterns with
s
similarproductstosupplementmorespecificanomalydetails. Weusepromptsalreadyemployed
inthepre-trainedvisual-linguisticdataset,e.g.,“black hole”and“white bubble”,toquerythe
desired regions. This approach reformulates the task of finding an anomaly region into locating
objectswithaspecificanomalystateexpression,whichismorestraightforwardtoutilizefoundation
modelsthanidentifying“anomaly”withinanobjectcontext.
BypromptingSAAwithanomalylanguagepromptsPL = {T ,T }derivedfromdomainexpert
a s
knowledge,wegeneratefineranomalyregioncandidatesRandcorrespondingconfidencescoresS.
4.1.2 AnomalyObjectPropertyasPrompt
Currentfoundationmodels[23,57]havelimitationswhenitcomestoqueryingobjectswithspecific
propertydescriptions,suchassizeorlocation,whichareimportantfordescribinganomalies,such
as“The small black hole on the left of the cable.”Toincorporatethiscriticalexpert
knowledge,weproposeusinganomalypropertypromptsformulatedasrulesratherthanlanguage.
Specifically,weconsiderthelocationandareaofanomalies.
AnomalyLocation. Accuratelocalizationofanomaliesplaysacriticalroleindistinguishingtrue
anomaliesfromfalsepositives. Typically,anomaliesareexpectedtobelocatedwithintheobjects
ofinterestduringinference. However,duetotheinfluenceofbackgroundcontext,anomaliesmay
occasionallyappearoutsidetheinspectedobjects. Totacklethischallenge,weleveragetheopen-
worlddetectioncapabilityoffoundationmodelstodeterminethelocationoftheinspectedobject.
Subsequently,wecalculatetheintersectionoverunion(IoU)betweenthepotentialanomalyregions
andtheinspectedobject. Byapplyinganexpert-derivedIoUthreshold,denotedasθ ,wefilter
IoU
outanomalycandidateswithIoUvaluesbelowthisthreshold. Thisprocessensuresthattheretained
anomalycandidatesaremorelikelytorepresenttrueanomalieslocatedwithintheinspectedobject.
AnomalyArea. Thesizeofananomaly,asreflectedbyitsarea,isalsoapropertythatcanprovide
usefulinformation. Ingeneral,anomaliesshouldbesmallerthanthesizeoftheinspectedobject.
Expertscanprovideasuitablethresholdvalueθ forthespecifictypeofanomalybeingconsidered.
area
Candidateswithareasunmatchedwithθ ·ObjectAreacanthenbefilteredout.
area
By combining the two property prompts PP = {θ ,θ }, we can filter the set of candidate
area IoU
regionsRtoobtainasubsetofselectedcandidatesRP withcorrespondingconfidencescoresSP
usingthefilterfunction(Filter),
RP,SP :=Filter(R,PP) (4)
4.2 PromptsDerivedfromTargetImageContext
Besidesincorporatingdomainexpertknowledge,wecanleveragetheinformationprovidedbythe
inputimageitselftoimprovetheaccuracyofanomalyregiondetection. Inthisregard,wepropose
twopromptsinducedbytheimagecontext.
4.2.1 AnomalySaliencyasPrompt
Predictionsgeneratedbyfoundationmodelslike[23]usingtheprompt“defect”canbeunreliable
due to the domain gap between pre-trained language-vision datasets [41] and targeted anomaly
segmentation datasets [4, 58]. To calibrate the confidence scores of individual predictions, we
proposeAnomalySaliencyPromptmimickinghumanintuition. Inspecific,humanscanrecognize
anomalyregionsbytheirdiscrepancywiththeirsurroundingregions[40],i.e.,visualsaliencycontains
valuableinformationindicatingtheanomalydegree. Hence,wecalculateasaliencymap(s)forthe
inputimagebycomputingtheaveragedistancesbetweenthecorrespondingpixelfeature(f)andits
N nearestneighbors,
1 (cid:88)
s := (1−(cid:104)f ,f(cid:105)) (5)
ij N ij
f∈Np(fij)
where(i,j)denotestothepixellocation,N (f )denotestotheN nearestneighborsofthecorre-
p ij
spondingpixel,and(cid:104)·,·(cid:105)referstothecosinesimilarity. Weusepre-trainedCNNsfromlarge-scale
imagedatasets[59]toextractimagefeatures,ensuringthedescriptivenessoffeatures. Thesaliency
6

mapindicateshowdifferentaregionisfromotherregions. ThesaliencypromptsPS aredefinedas
theexponentialaveragesaliencyvaluewithinthecorrespondingregionmasks,
(cid:40) (cid:80) (cid:41)
r s
PS := exp( ij ij ij ) | r∈RP (6)
(cid:80)
r
ij ij
The saliency prompts provide reliable indications of the confidence of anomaly regions. These
prompts are employed to recalibrate the confidence scores generated by the foundation models,
yieldingnewrescaledscoresSS basedontheanomalysaliencypromptsPS. Theserescaledscores
provideacombinedmeasurethattakesintoaccountboththeconfidencederivedfromthefoundation
modelsandthesaliencyoftheregioncandidate. Theprocessisformulatedasfollows,
SS := (cid:8) p·s | p∈PS,s∈SP(cid:9) (7)
4.2.2 AnomalyConfidenceasPrompt
Typically,thenumberofanomalyregionsinaninspectedobjectislimited. Therefore,wepropose
anomalyconfidencepromptsPC toidentifytheK candidateswiththehighestconfidencescores
basedontheimagecontentandusetheiraveragevaluesforfinalanomalyregiondetection. Thisis
achievedbyselectingthetopK candidateregionsbasedontheircorrespondingconfidencescores,as
showninthefollowing,
RC,SC :=Top (RP,SS) (8)
K
DenoteasingleregionanditscorrespondingscoreasrC andsC,wethenusetheseK candidate
regionstoestimatethefinalanomalymap,
(cid:80) rC ·sC
A := rC∈RC ij (9)
ij (cid:80) rC
rC∈RC ij
Withtheproposedhybridprompts(PL,PP,PS,andPC),SAAisregularizedinourfinalframework,
i.e.,SegmentAnyAnomaly+(SAA+),whichmakesmorereliableanomalypredictions.
5 Experiments
Inthissection, wefirstassesstheperformanceofSAA/SAA+onseveralanomalysegmentation
benchmarks. Then,weextensivelystudytheeffectivenessofindividualhybridprompts.
5.1 ExperimentalSetup
Datasets. We leverage four datasets with pixel-level annotations.: VisA [58], MVTec-AD [4],
KSDD2[60],andMTD[61]. VisAandMVTec-ADcompriseavarietyofobjectsubsets,e.g.,circuit
boards,whileKSDD2andMTDarecomprisedoftextureanomalies. Insummary,wecategorizethe
subsetsofallofthesedatasetsintotexturewhichtypicallyexhibitsimilarpatternswithinasingle
image(e.g.,carpets),andobjectwhichincludesmorediversedistribution(e.g.,candles).
EvaluationMetrics. ZSASperformanceisevaluatedbasedontwometrics: (I)max-F1-pixel(F )
p
[25],whichmeasurestheF1-scoreforpixel-wisesegmentationattheoptimalthreshold;(II)max-F1-
region(F ),whichisproposedinthispapertomitigatethebiastowardslargedefectsobservedwith
r
max-F1-pixel[4]. Specifically,wecomputetheF1-scoreforregion-wisesegmentationattheoptimal
threshold,consideringapredictionpositiveiftheoverlappingvalueexceeds0.6.
ImplementationDetails. WeadopttheofficialimplementationsofGroundingDINO1andSegment
AnythingModel2 toconstructthevanillabaseline(SAA).Detailsaboutthepromptsderivedfrom
domainexpertknowledgeareexplainedinthesupplementarymaterial. Forthesaliencypromptsin-
ducedfromimagecontent,weutilizetheWideResNet50[62]network,pre-trainedonImageNet[59],
and set N = 400 in line with prior studies [40]. For anomaly confidence prompts, we set the
hyperparameterK as5bydefault. Inputimagesarefixedataresolutionof400×400forevaluation.
1https://github.com/IDEA-Research/GroundingDINO
2https://github.com/facebookresearch/segment-anything
7

Table1:QualitativecomparisonsbetweenSAA+andotherconcurrentmethodsonzero-shotanomaly
segmentation. Bestscoresarehighlightedinbold. Thesecondbestscoresarealsounderlined.
|               | PerDataset    |             | PerDefectType  |       |
| ------------- | ------------- | ----------- | -------------- | ----- |
| Metric Method |               |             |                | Total |
|               | VisA MVTec-AD | KSDD2 MTD   | Texture Object |       |
| WinClip[25]   | 14.82 31.65   | - -         | - 20.93        | -     |
| ClipSeg[24]   | 14.32 25.42   | 34.27 9.39  | 27.75 18.30    | 20.58 |
| F UTAD[40]    | 6.95 23.48    | 22.53 11.37 | 29.13 12.07    | 16.19 |
p
| SAA         | 12.76 23.44 | 8.79 14.78  | 20.94 17.35 | 18.22 |
| ----------- | ----------- | ----------- | ----------- | ----- |
| SAA+        | 27.07 39.40 | 59.19 35.40 | 53.79 28.82 | 34.85 |
| ClipSeg[24] | 5.65 19.68  | 9.05 6.55   | 21.37 10.41 | 13.06 |
| UTAD[40]    | 5.32 17.53  | 3.56 2.95   | 16.38 9.94  | 11.49 |
| F SAA       | 4.83 32.49  | 16.40 10.63 | 40.31 13.19 | 19.74 |
r
| SAA+ | 14.46 49.67 | 39.34 30.27 | 60.40 25.70 | 34.07 |
| ---- | ----------- | ----------- | ----------- | ----- |
Figure3: Qualitativecomparisonsonzero-shotanomalysegmentationforClipSeg[24],UTAD[40],
SAA,andSAA+onfourdatasets,i.e.,VisA[58],MVTec-AD[4],KSDD2[60],andMTD[61]
5.2 MainResults
MethodsforComparison. Wecompareourfinalmodel,i.e.,SegmentAnyAnomaly+(SAA+)
withseveralconcurrentstate-of-the-artmethods,includingWinClip[25],UTAD[40],ClipSeg[24],
andourvanillabaseline(SAA).ForWinClip,wereportitsofficialresultsonVisAandMVTec-AD.
For the other three methods, we use official implementations and adapt them to the ZSAS task.
Notably,asallmethodsrequirenotrainingprocess,theirperformanceisstablewithavarianceof
±0.00.
QuantitativeResults: AsisshowninTable1,SAA+methodoutperformsothermethodsinbothF
p
andF byasignificantmargin. AlthoughWinClip[25],ClipSeg[24],andSAAalsousefoundation
r
models,SAA+betterunleashthecapacityoffoundationmodelsandadaptsthemtotackleZSAS.The
remarkableperformanceofSAA+meetstheexpectationtosegmentanyanomalywithouttraining.
QualitativeResults: Fig. 3presentsqualitativecomparisonsbetweenSAA+andpreviouscompeti-
tivemethods,whereSAA+achievesbetterperformance. Moreover,thevisualizationshowsSAA+
| iscapableofdetectingtextureanomalies,e.g. | smallscratchesontheleather. |     |     |     |
| ----------------------------------------- | --------------------------- | --- | --- | --- |
8

Table 2: Ablation study on the proposed hy-
bridprompts,includinglanguageprompt(PL),
object property prompt (PP), saliency prompt
(PS), and confidence prompt (PC). The best
scoresarehighlightedinbold.
Metric ModelVariants TextureObjectTotal
w/oT &T 50.30 24.79 30.95
a s
w/oPLw/oT 51.15 25.88 31.80
a
w/oT 53.51 26.55 33.06
s
w/oPP 21.83 21.40 21.50 Figure4: Effectsofdisabling(w/o)andabling
F
p w/oPS 50.58 24.72 30.96
(w/)prompts(PS)ofsaliencymaps(s)onthe
finalanomalysegmentation.
w/oPC 50.41 27.99 34.13
fullmodel(SAA+) 53.79 28.82 34.85
w/oT &T 50.58 22.36 29.17
a s
w/oPLw/oT 55.26 20.28 28.72
a
w/oT 54.21 23.13 30.64
s
w/oPP 33.94 20.99 24.11
F
r w/oPS 57.66 24.36 32.39
w/oPC 53.65 25.18 32.05
Figure5: Sensitivityanalysisofhyperparameter
fullmodel(SAA+) 60.40 25.70 34.07 K ofconfidenceprompts(PC).
5.3 Ablationstudy
InTable2,weperformcomponent-wiseanalysistoablatespecificpromptdesignsinourframework.
Language prompt (PL). Table 2 verifies the effectiveness of language prompts derived from
domain expert knowledge (+3.90% in F and +4.90% in F ). Then, we dig into the efficacy of
p r
T and T , which clearly indicate that both the general description and the specifically designed
a s
descriptionforanomaliescanachievereasonableperformance. Moreover,theircombinationcan
make a synergy, enhancing anomaly segmentation performance. The improvement of PL helps
unlocklanguage-drivenregiondetectioncapacityofcurrentfoundationmodels[23,19].
Propertyprompt(PP). Apartfromtheimprovementintheoverallperformance,propertyprompts
bring dramatic improvements (from 21.83% to 53.79% in F ) on texture categories, thanks to
p
thefilteringmechanismwhichfiltersoutasignificantnumberoffalselydetectedanomalyregion
candidatesviahigh-levelcharacteristics,e.g.,locationandareaofthetargetimage.
Saliencyprompt(PS). Table2providesclearevidenceoftheefficacyofPS onanomalysegmenta-
tion. Thisisbecauseregionsalienciescanaccuratelydescribethedegreeofdeviationofaregion
fromitssurroundings. InFig. 4,weshowcasethequalitativeimpactofPS onanomalysegmentation,
whichillustratesvisualsaliencymapscanhelphighlightabnormalregions,i.e.,whichshowshigher
saliencyvaluescomparedtootherregions. ByincorporatingPS tocalibratetheconfidencescores,
moreprecisesegmentationresultscanbeachieved. Forexample,theuseofPS enablestheeffective
localizationofthecrackedregionofhazelnutandtheoverlongwickoncandles.
Confidence prompt (PC). With the incorporation of anomaly confidence prompts, we limit the
numberofanomalyregions,whicheffectivelyreducesfalsepositives,leadingto0.72%F average
p
improvementsacrossallcategories,asshowninTable2. TheinfluenceofthehyperparameterK
inPC isillustratedinFig. 5. ThefigureshowsthatperformanceinitiallyincreasesasK improves,
asmoreanomalyregionsareaccuratelydetected. However, whenK exceedsacertainthreshold
(aroundK =5),theperformancedropsslightlyasmoreregionsarewronglyidentifiedasabnormal.
ThebestresultsareobtainedataroundK =5,withanaverageF of34.85%acrossallcategories.
p
9

6 Conclusion
Inthiswork,weexplorehowtosegmentanyanomalywithoutanyfurthertrainingbyunleashingthe
fullpowerofmodernfoundationmodels.Weowethestruggleofadaptingfoundationmodelassembly
toanomalysegmentationtothepromptdesign,whichisthekeytocontrollingthefunctionofoff-the-
shelffoundationmodels. Thus,weproposeanovelframework,i.e.,SegmentAnyAnomaly+,to
leveragehybridpromptsderivedfrombothexpertknowledgeandtargetimagecontexttoregularize
foundationmodelsfreeoftraining. Finally,wesuccessfullyadaptmultiplefoundationmodelsto
tacklezero-shotanomalysegmentation,achievingnewSoTAresultsonseveralbenchmarks. We
hopeourworkcanshedlightonthedesignoflabel-freemodeladaptationforanomalysegmentation.
Limitations. Duetothecomputationrestriction,wecurrentlydonottestourmethodonmorelarge-
scalefoundationmodels. Wehavefinishedtheexplorationofourmethodologywithrepresentative
foundationmodels,andwewillexplorethescalingeffectofthemodelsinthefuture.
References
[1] YunkangCao,XiaohaoXu,ZhaogeLiu,andWeimingShen. Collaborativediscrepancyoptimizationfor
reliableimageanomalylocalization. IEEETransactionsonIndustrialInformatics,pages1–10,2023.
[2] QianWan,LiangGao,XinyuLi,andLongWen. Industrialimageanomalylocalizationbasedongaussian
clusteringofpretrainedfeature. IEEETransactionsonIndustrialElectronics,69(6):6182–6192,2021.
[3] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler.
Towardstotalrecallinindustrialanomalydetection. InProceedingsoftheIEEE/CVFConferenceon
ComputerVisionandPatternRecognition,pages14318–14328,2022.
[4] PaulBergmann,MichaelFauser,DavidSattlegger,andCarstenSteger. MVTecAD–Acomprehensive
real-worlddatasetforunsupervisedanomalydetection. InProceedingsoftheIEEE/CVFconferenceon
ComputerVisionandPatternRecognition,pages9592–9600,2019.
[5] PaulBergmann,MichaelFauser,DavidSattlegger,andCarstenSteger. Uninformedstudents: Student-
teacher anomaly detection with discriminative latent embeddings. In Proceedings of the IEEE/CVF
ConferenceonComputerVisionandPatternRecognition,pages4183–4192,2020.
[6] ChristophBaur,StefanDenner,BenediktWiestler,NassirNavab,andShadiAlbarqouni. Autoencodersfor
unsupervisedanomalysegmentationinbrainmrimages:acomparativestudy. MedicalImageAnalysis,
69:101952,2021.
[7] KangZhou, YutingXiao, JianlongYang, JunCheng, WenLiu, WeixinLuo, ZaiwangGu, JiangLiu,
and Shenghua Gao. Encoding structure-texture relation with p-net for anomaly detection in retinal
images. InComputerVision–ECCV2020:16thEuropeanConference,Glasgow,UK,August23–28,2020,
Proceedings,PartXX16,pages360–377.Springer,2020.
[8] JinleiHou,YingyingZhang,QiaoyongZhong,DiXie,ShiliangPu,andHongZhou. Divide-and-assemble:
Learningblock-wisememoryforunsupervisedanomalydetection. InProceedingsoftheIEEE/CVF
InternationalConferenceonComputerVision,pages8791–8800,2021.
[9] VitjanZavrtanik,MatejKristan,andDanijelSkocˇaj. DRAEM–Adiscriminativelytrainedreconstruction
embeddingforsurfaceanomalydetection. InProceedingsoftheIEEE/CVFInternationalConferenceon
ComputerVision,pages8330–8339,2021.
[10] TakashiMatsubara,KazukiSato,KentaHama,RyosukeTachibana,andKuniakiUehara. Deepgenerative
modelusingunregularizedscoreforanomalydetectionwithheterogeneouscomplexity.IEEETransactions
onCybernetics,52(6):5161–5173,2020.
[11] XudongYan,HuaidongZhang,XuemiaoXu,XiaoweiHu,andPheng-AnnHeng. Learningsemantic
contextfromnormalsamplesforunsupervisedanomalydetection. InProceedingsoftheAAAIConference
onArtificialIntelligence,volume35,pages3110–3118,2021.
[12] JielinJiang,JialeZhu,MuhammadBilal,YanCui,NeerajKumar,RuihanDou,FengSu,andXiaolong
Xu. Maskedswintransformerunetforindustrialanomalydetection. IEEETransactionsonIndustrial
Informatics,19(2):2200–2209,2022.
[13] JihunYiandSungrohYoon. PatchSVDD:Patch-levelSVDDforanomalydetectionandsegmentation. In
ProceedingsoftheAsianConferenceonComputerVision,2020.
[14] FabioValerioMassoli,FabrizioFalchi,AlperenKantarci,S¸eymanurAkti,HazimKemalEkenel,and
GiuseppeAmato. Mocca:Multilayerone-classclassificationforanomalydetection. IEEETransactionson
NeuralNetworksandLearningSystems,33(6):2313–2323,2021.
10

[15] KihyukSohn,Chun-LiangLi,JinsungYoon,MinhoJin,andTomasPfister. Learningandevaluating
representationsfordeepone-classclassification. InInternationalConferenceonLearningRepresentations,
2020.
[16] YunkangCao,XiaohaoXu,andWeimingShen. Complementarypseudomultimodalfeatureforpoint
cloudanomalydetection. arXivpreprintarXiv:2303.13194,2023.
[17] YueWang, JinlongPeng, JiangningZhang, RanYi, YabiaoWang, andChengjieWang. Multimodal
industrialanomalydetectionviahybridfusion. In2023IEEE/CVFConferenceonComputerVisionand
PatternRecognition(CVPR),2023.
[18] XiJiang,JianlinLiu,JinbaoWang,QiangNie,KaiWu,YongLiu,ChengjieWang,andFengZheng.
SoftPatch:Unsupervisedanomalydetectionwithnoisydata. InAdvancesinneuralinformationprocessing
systems,2022.
[19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
Xiao, SpencerWhitehead, AlexanderCBerg, Wan-YenLo, etal. Segmentanything. arXivpreprint
arXiv:2304.02643,2023.
[20] AlecRadford,JongWookKim,ChrisHallacy,AdityaRamesh,GabrielGoh,SandhiniAgarwal,Girish
Sastry,AmandaAskell,PamelaMishkin,JackClark,etal. Learningtransferablevisualmodelsfrom
naturallanguagesupervision.InInternationalConferenceonMachineLearning,pages8748–8763.PMLR,
2021.
[21] Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, and Steven CH Hoi. Align and prompt:
Video-and-languagepre-trainingwithentityprompts. InProceedingsoftheIEEE/CVFConferenceon
ComputerVisionandPatternRecognition,pages4953–4963,2022.
[22] RishiBommasani,DrewAHudson,EhsanAdeli,RussAltman,SimranArora,SydneyvonArx,MichaelS
Bernstein,JeannetteBohg,AntoineBosselut,EmmaBrunskill,etal. Ontheopportunitiesandrisksof
foundationmodels. arXivpreprintarXiv:2108.07258,2021.
[23] ShilongLiu,ZhaoyangZeng,TianheRen,FengLi,HaoZhang,JieYang,ChunyuanLi,JianweiYang,
HangSu,JunZhu,etal. Groundingdino:Marryingdinowithgroundedpre-trainingforopen-setobject
detection. arXivpreprintarXiv:2303.05499,2023.
[24] TimoLüddeckeandAlexanderEcker. Imagesegmentationusingtextandimageprompts. InProceedings
oftheIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages7086–7096,2022.
[25] JongheonJeong,YangZou,TaewanKim,DongqingZhang,AvinashRavichandran,andOnkarDabeer.
Winclip:Zero-/few-shotanomalyclassificationandsegmentation. arXivpreprintarXiv:2303.14814,2023.
[26] XiaohaoXu,JingluWang,XiangMing,andYanLu. Towardsrobustvideoobjectsegmentationwith
adaptiveobjectcalibration. InProceedingsofthe30thACMInternationalConferenceonMultimedia,
pages1–10,2022.
[27] XiaohaoXu,JingluWang,XiaoLi,andYanLu. Reliablepropagation-correctionmodulationforvideo
objectsegmentation. InProceedingsoftheAAAIConferenceonArtificialIntelligence,pages2946–2954,
2022.
[28] RoniPaiss,ArielEphrat,OmerTov,ShiranZada,InbarMosseri,MichalIrani,andTaliDekel. Teaching
cliptocounttoten. arXivpreprintarXiv:2302.12066,2023.
[29] XiangLi,JingluWang,XiaohaoXu,XiaoLi,YanLu,andBhikshaRaj. Rˆ2vos:Robustreferringvideo
objectsegmentationviarelationalmultimodalcycleconsistency. arXivpreprintarXiv:2207.01203,2022.
[30] MohammadrezaSalehi,NioushaSadjadi,SorooshBaselizadeh,MohammadHRohban,andHamidR
Rabiee. Multiresolutionknowledgedistillationforanomalydetection. InProceedingsoftheIEEE/CVF
ConferenceonComputerVisionandPatternRecognition,pages14902–14912,2021.
[31] GuodongWang,ShuminHan,ErruiDing,andDiHuang. Student-teacherfeaturepyramidmatchingfor
anomalydetection. arXivpreprintarXiv:2103.04257,2021.
[32] HanqiuDengandXingyuLi. Anomalydetectionviareversedistillationfromone-classembedding. In
ProceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages9737–9746,
2022.
[33] YunkangCao,QianWan,WeimingShen,andLiangGao. Informativeknowledgedistillationforimage
anomalysegmentation. Knowledge-BasedSystems,248:108846,2022.
[34] YunkangCao,YananSong,XiaohaoXu,ShuyaLi,YuhaoYu,YihengZhang,andWeimingShen. Semi-
supervisedknowledgedistillationfortinydefectdetection. In2022IEEE25thInternationalConference
onComputerSupportedCooperativeWorkinDesign(CSCWD),pages1010–1015,2022.
[35] QianWan,LiangGao,XinyuLi,andLongWen.Unsupervisedimageanomalydetectionandsegmentation
basedonpre-trainedfeaturemapping. IEEETransactionsonIndustrialInformatics,2022.
11

[36] QianWan,YunkangCao,LiangGao,WeimingShen,andXinyuLi. Positionencodingenhancedfeature
mappingforimageanomalydetection.In2022IEEE18thInternationalConferenceonAutomationScience
andEngineering(CASE),pages876–881.IEEE,2022.
[37] AmrMNagyandLászlóCzúni.Zero-shotlearningandclassificationofsteelsurfacedefects.InFourteenth
InternationalConferenceonMachineVision(ICMV2021),volume12084,pages386–394.SPIE,2022.
[38] JiahuiLiu,XiaojuanQi,SongzhiSu,TonyPrescott,andLiSun. Zero-shotanomalousobjectdetection
usingunsupervisedmetriclearning. In2021IEEE/RSJInternationalConferenceonIntelligentRobotsand
Systems(IROS2021)Proceedings.Sheffield,2021.
[39] AdínRamírezRivera,AdilKhan,ImadEddineIbrahimBekkouch,andTaimoorShakeelSheikh. Anomaly
detectionbasedonzero-shotoutliersynthesisandhierarchicalfeaturedistillation. IEEETransactionson
NeuralNetworksandLearningSystems,33(1):281–291,2020.
[40] ToshimichiAota,LloydTehTzerTong,andTakayukiOkatani. Zero-shotversusmany-shot:Unsupervised
texture anomaly detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of
ComputerVision,pages5564–5572,2023.
[41] ChristophSchuhmann,RobertKaczmarczyk,AranKomatsuzaki,AarushKatta,RichardVencu,Romain
Beaumont,JeniaJitsev,TheoCoombes,andClaytonMullis. Laion-400m:Opendatasetofclip-filtered
400millionimage-textpairs. InNeurIPSWorkshopDatacentricAI.JülichSupercomputingCenter,2021.
[42] JunnanLi,RamprasaathSelvaraju,AkhileshGotmare,ShafiqJoty,CaimingXiong,andStevenChuHong
Hoi. Alignbeforefuse: Visionandlanguagerepresentationlearningwithmomentumdistillation. In
Advancesinneuralinformationprocessingsystems,volume34,pages9694–9705,2021.
[43] JiasenLu,ChristopherClark,RowanZellers,RoozbehMottaghi,andAniruddhaKembhavi. Unified-IO:A
unifiedmodelforvision,language,andmulti-modaltasks. arXivpreprintarXiv:2206.08916,2022.
[44] PengWang,AnYang,RuiMen,JunyangLin,ShuaiBai,ZhikangLi,JianxinMa,ChangZhou,Jingren
Zhou,andHongxiaYang. Unifyingarchitectures,tasks,andmodalitiesthroughasimplesequence-to-
sequencelearningframework. arXivpreprintarXiv:2202.03052,2022.
[45] YongmingRao,WenliangZhao,GuangyiChen,YansongTang,ZhengZhu,GuanHuang,JieZhou,and
JiwenLu. Denseclip:Language-guideddensepredictionwithcontext-awareprompting. InProceedingsof
theIEEEConferenceonComputerVisionandPatternRecognition(CVPR),2022.
[46] YiwuZhong,JianweiYang,PengchuanZhang,ChunyuanLi,NoelCodella,LiunianHaroldLi,Luowei
Zhou,XiyangDai,LuYuan,YinLi,etal. Regionclip: Region-basedlanguage-imagepretraining. In
ProceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages16793–
16803,2022.
[47] ChongZhou,ChenChangeLoy,andBoDai. Extractfreedenselabelsfromclip. InComputerVision–
ECCV2022:17thEuropeanConference,TelAviv,Israel,October23–27,2022,Proceedings,PartXXVIII,
pages696–712.Springer,2022.
[48] KaiyangZhou,JingkangYang,ChenChangeLoy,andZiweiLiu. Conditionalpromptlearningforvision-
languagemodels. In2022IEEE/CVFConferenceonComputerVisionandPatternRecognition(CVPR),
2022.
[49] ChenJu,TengdaHan,KunhaoZheng,YaZhang,andWeidiXie. Promptingvisual-languagemodelsfor
efficientvideounderstanding. InComputerVision–ECCV2022: 17thEuropeanConference,TelAviv,
Israel,October23–27,2022,Proceedings,PartXXXV,pages105–124.Springer,2022.
[50] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and
Ser-NamLim. Visualprompttuning. InComputerVision–ECCV2022:17thEuropeanConference,Tel
Aviv,Israel,October23–27,2022,Proceedings,PartXXXIII,pages709–727.Springer,2022.
[51] HyojinBahng,AliJahanian,SwamiSankaranarayanan,andPhillipIsola. Exploringvisualpromptsfor
adaptinglarge-scalemodels. arXivpreprintarXiv:2203.17274,1(3):4,2022.
[52] YuhangZang,WeiLi,KaiyangZhou,ChenHuang,andChenChangeLoy. Unifiedvisionandlanguage
promptlearning. arXivpreprintarXiv:2210.07225,2022.
[53] ShengShen, ShijiaYang, TianjunZhang, BohanZhai, JosephEGonzalez, KurtKeutzer, andTrevor
Darrell. Multitaskvision-languageprompttuning. arXivpreprintarXiv:2211.11720,2022.
[54] KaiyangZhou,JingkangYang,ChenChangeLoy,andZiweiLiu. Learningtopromptforvision-language
models. IntJComputVis,130(9):2337–2348,2022.
[55] AleksandarShtedritski,ChristianRupprecht,andAndreaVedaldi. Whatdoesclipknowaboutaredcircle?
visualpromptengineeringforvlms. arXivpreprintarXiv:2304.06712,2023.
[56] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit,
andNeilHoulsby. Animageisworth16x16words: Transformersforimagerecognitionatscale. In
InternationalConferenceonLearningRepresentations,2021.
12

[57] LiunianHaroldLi,PengchuanZhang,HaotianZhang,JianweiYang,ChunyuanLi,YiwuZhong,Lijuan
Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In
ProceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecognition,pages10965–
10975,2022.
[58] YangZou,JongheonJeong,LathaPemula,DongqingZhang,andOnkarDabeer. SPot-the-Difference
self-supervisedpre-trainingforanomalydetectionandsegmentation. InProceedingsoftheEuropean
ConferenceonComputerVision,2022.
[59] GeoffreyEHinton,AlexKrizhevsky,andIlyaSutskever. ImageNetclassificationwithdeepconvolutional
neuralnetworks. AdvancesinNeuralInformationProcessingSystems,25(1106-1114):1,2012.
[60] JakobBožicˇ,DomenTabernik,andDanijelSkocˇaj. Mixedsupervisionforsurface-defectdetection:From
weaklytofullysupervisedlearning. ComputersinIndustry,129:103459,2021.
[61] YibinHuang,CongyingQiu,YueGuo,XiaonanWang,andKuiYuan. Surfacedefectsaliencyofmagnetic
tile. In2018IEEE14thInternationalConferenceonAutomationScienceandEngineering(CASE),pages
612–617,2018.
[62] SergeyZagoruykoandNikosKomodakis.Wideresidualnetworks.InEdwinR.HancockRichardC.Wilson
andWilliamA.P.Smith,editors,ProceedingsoftheBritishMachineVisionConference(BMVC),pages
87.1–87.12.BMVAPress,September2016.
13
