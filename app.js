
const screens=["landing","acquisition","quality","preprocess","analysis","xai","review","report"];
let current=0,selectedFile=null,latestResult=null;
const $=id=>document.getElementById(id), navDots=$("navDots");

screens.forEach((_,i)=>{const d=document.createElement("div");d.className="nav-dot"+(i===0?" active":"");d.onclick=()=>goTo(i);navDots.appendChild(d);});
function goTo(i){current=Math.max(0,Math.min(screens.length-1,i));screens.forEach((id,j)=>$(id).classList.toggle("active-screen",j===current));[...navDots.children].forEach((d,j)=>d.classList.toggle("active",j===current));$("backBtn").disabled=current===0;$("nextBtn").disabled=current===screens.length-1;window.scrollTo({top:0,behavior:"smooth"});}
$("startBtn").onclick=()=>goTo(1);
$("browseBtn").onclick=e=>{e.stopPropagation();$("imageInput").click();};
$("uploadZone").onclick=()=>$("imageInput").click();
$("imageInput").onchange=e=>e.target.files[0]&&loadImage(e.target.files[0]);
["dragenter","dragover"].forEach(ev=>$("uploadZone").addEventListener(ev,e=>{e.preventDefault();$("uploadZone").classList.add("drag");}));
["dragleave","drop"].forEach(ev=>$("uploadZone").addEventListener(ev,e=>{e.preventDefault();$("uploadZone").classList.remove("drag");}));
$("uploadZone").addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f&&f.type.startsWith("image/"))loadImage(f);});

function loadImage(f){
 selectedFile=f;$("fileName").textContent=f.name;const u=URL.createObjectURL(f);
 $("imagePreview").innerHTML=`<img class="preview-img" src="${u}" alt="Fundus">`;
 $("qualityImage").style.backgroundImage=`url("${u}")`;
 ["originalXai","preOriginal"].forEach(id=>{const x=$(id);x.style.backgroundImage=`url("${u}")`;x.style.backgroundSize="contain";x.style.backgroundRepeat="no-repeat";x.style.backgroundPosition="center";});
 $("analyzeBtn").disabled=false;
}
function pending(){
 ["focusMetric","illumMetric","visibilityMetric","gradabilityMetric"].forEach(id=>$(id).textContent="Analyzing...");
 $("qualityState").textContent="ANALYZING IMAGE";$("qualitySub").textContent="MATLAB integrated pipeline running...";
 $("gatewayTitle").textContent="FUNDUS-DOMAIN SAFETY GATEWAY";$("gatewaySub").textContent="Checking whether input resembles a retinal fundus photograph";$("gatewayScore").textContent="Fundus-likeness: —";
}
$("analyzeBtn").onclick=async()=>{
 if(!selectedFile)return;goTo(2);pending();
 const fd=new FormData();fd.append("image",selectedFile);
 try{
  const r=await fetch("/api/analyze",{method:"POST",body:fd});const d=await r.json();
  if(!r.ok)throw new Error(d.error||"Analysis failed");
  latestResult=d;apply(d);
  if(d.gateway_pass) setTimeout(()=>goTo(3),1200);
 }catch(e){$("qualityState").textContent="ANALYSIS FAILED";$("qualitySub").textContent=e.message;$("gatewayTitle").textContent="CHECK MATLAB BRIDGE";console.error(e);}
};
function image(id,url){if(!url)return;const x=$(id);x.innerHTML="";x.style.backgroundImage=`url("${url}?t=${Date.now()}")`;x.style.backgroundSize="contain";x.style.backgroundRepeat="no-repeat";x.style.backgroundPosition="center";}
function apply(d){
 $("qualityState").textContent=d.quality_result||"—";$("qualitySub").textContent=d.quality_action||"";
 $("focusMetric").textContent=num(d.focus_score,6);$("illumMetric").textContent=num(d.brightness_score,4);
 $("visibilityMetric").textContent=num(d.contrast_score,4);$("gradabilityMetric").textContent=d.quality_result==="UNGRADABLE"?"REJECTED":"ACCEPTED";
 if(d.clahe_url){image("claheView",d.clahe_url);}
 if(d.original_url){image("preOriginal",d.original_url);}
 $("origPred").textContent=d.dr_prediction||"—";
 $("clahePred").textContent=d.dr_prediction_clahe||"—";
 $("origConf").textContent=pct01(d.dr_confidence);
 $("claheConf").textContent=pct01(d.dr_confidence_clahe);
 $("preprocessStability").textContent=d.clahe_stability==="STABLE"?"PREDICTION STABLE ✓":(d.clahe_stability==="UNSTABLE"?"PREDICTION CHANGED — HUMAN REVIEW":"STABILITY CHECK");
 $("preprocessStabilityReason").textContent=d.clahe_stability==="STABLE"?"Original and CLAHE predictions agree.":"Original and CLAHE predictions are compared as a safety check.";
 $("gatewayTitle").textContent=d.gateway_pass?"FUNDUS GATEWAY PASSED":"INFERENCE BLOCKED";
 $("gatewaySub").textContent=d.gateway_reason||"—";$("gatewayScore").textContent=`Fundus-likeness: ${pct01(d.gateway_score)}`;

 if(!d.gateway_pass){
   $("resultGrade").textContent="NOT RUN";$("resultConfidence").textContent="—";$("resultReferable").textContent="BLOCKED";$("resultReview").textContent="RECAPTURE / HUMAN REVIEW";
   $("finalQuality").textContent=d.quality_result||"—";$("finalGrade").textContent="NOT GRADED";$("finalConfidence").textContent="—";$("finalReferral").textContent="INFERENCE BLOCKED";$("finalReview").textContent="REQUIRED";$("finalRecommendation").textContent=d.gateway_reason||"Recapture or human assessment required.";
   return;
 }
 $("resultGrade").textContent=d.dr_prediction||"—";$("resultConfidence").textContent=pct01(d.dr_confidence);$("resultReferable").textContent=d.referral_result||"—";$("resultReview").textContent=d.prediction_safety==="BLOCKED"?"MANDATORY HUMAN REVIEW":"HUMAN VALIDATION";
 $("reviewGrade").textContent=d.dr_prediction||"—";$("reviewConfidence").textContent=pct01(d.dr_confidence);$("reviewReferral").textContent=d.referral_result||"—";
 $("finalQuality").textContent=d.quality_result||"—";$("finalGrade").textContent=d.dr_prediction||"—";$("finalConfidence").textContent=pct01(d.dr_confidence);$("finalReferral").textContent=d.referral_result||"—";$("finalRecommendation").textContent=d.referral_action||"—";
 $("vesselArea").textContent=(d.vessel_area_percent??0).toFixed(4)+"%";$("lesionArea").textContent=(d.lesion_foreground_area_percent??0).toFixed(6)+"%";$("claheStability").textContent=d.clahe_stability||"—";$("predictionSafety").textContent=d.prediction_safety||"—";
 image("gradcamXai",d.gradcam_overlay_url);image("vesselXai",d.vessel_overlay_url);image("lesionXai",d.lesion_overlay_url);image("combinedXai",d.combined_overlay_url);
 if(d.landmarks_url){image("landmarksXai",d.landmarks_url);$("landmarksNote").textContent="Estimated optic disc and experimental fovea candidate. Not clinically validated.";}
 else{$("landmarksXai").style.backgroundImage="none";$("landmarksNote").textContent="Experimental landmarks unavailable for this image. DR screening results are unaffected.";}
}
const num=(x,n)=>typeof x==="number"?x.toFixed(n):"—", pct01=x=>typeof x==="number"?(100*x).toFixed(2)+"%":"—";
$("acceptReview").onclick=async()=>{await saveReview("ACCEPTED_BY_HUMAN","Human reviewer accepted the AI screening result.");$("reviewStatus").textContent="✓ AI result accepted by human reviewer";$("reviewStatus").style.color="#15966b";$("finalReview").textContent="AI RESULT ACCEPTED";};
$("overrideReview").onclick=async()=>{const g=prompt("Enter clinician final grade / note:","Specialist review required");if(g){await saveReview("OVERRIDDEN_BY_HUMAN",g);$("reviewStatus").textContent="Reviewer override recorded: "+g;$("reviewStatus").style.color="#1769e0";$("finalReview").textContent="REVIEWER OVERRIDE";}};
$("recaptureReview").onclick=async()=>{await saveReview("RECAPTURE_REQUESTED","Acquire a new gradable fundus image.");$("reviewStatus").textContent="Image recapture requested";$("reviewStatus").style.color="#d94b5d";$("finalReview").textContent="RECAPTURE REQUESTED";$("finalRecommendation").textContent="Acquire a new gradable fundus image before final interpretation.";};
async function saveReview(decision,note){if(!latestResult?.case_id)return;try{await fetch(`/api/review/${latestResult.case_id}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision,note})});}catch(e){console.error(e);}}
$("backBtn").onclick=()=>goTo(current-1);$("nextBtn").onclick=()=>goTo(current+1);
$("newCaseBtn").onclick=()=>location.reload();
goTo(0);


/* Frontend-only localization and theme. Clinical model outputs are never modified. */
const UI_TRANSLATIONS = {
  "AI-assisted diabetic retinopathy screening":["मधुमेह रेटिनोपैथी की AI-सहायता प्राप्त स्क्रीनिंग","AI உதவியுடன் நீரிழிவு விழித்திரை நோய் பரிசோதனை"],
  "System ready":["सिस्टम तैयार है","அமைப்பு தயார்"],
  "RURAL INDIA":["ग्रामीण भारत","கிராமப்புற இந்தியா"],
  "SAFE • EXPLAINABLE • HUMAN-IN-THE-LOOP":["सुरक्षित • व्याख्यात्मक • मानवीय समीक्षा","பாதுகாப்பு • விளக்கத்தன்மை • மனித மதிப்பாய்வு"],
  "Trust the image.":["छवि पर भरोसा करें।","படத்தை நம்புங்கள்."],
  "Verify the AI.":["AI की जाँच करें।","AI-ஐச் சரிபாருங்கள்."],
  "Protect the patient.":["रोगी की सुरक्षा करें।","நோயாளியைப் பாதுகாப்போம்."],
  "Start screening":["स्क्रीनिंग शुरू करें","பரிசோதனையைத் தொடங்கு"],
  "Image quality":["छवि की गुणवत्ता","படத் தரம்"],
  "AI assessment":["AI मूल्यांकन","AI மதிப்பீடு"],
  "Human review":["मानवीय समीक्षा","மனித மதிப்பாய்வு"],
  "RETINAL AI":["रेटिनल AI","விழித்திரை AI"],
  "screening ready":["स्क्रीनिंग के लिए तैयार","பரிசோதனைக்குத் தயார்"],
  "Safety first":["सुरक्षा पहले","பாதுகாப்பு முதலில்"],
  "Quality gate before AI":["AI से पहले गुणवत्ता जाँच","AI-க்கு முன் தரச் சோதனை"],
  "Explainable":["व्याख्यात्मक","விளக்கத்தன்மை"],
  "Model-focused visual evidence":["मॉडल का दृश्य साक्ष्य","மாதிரியின் காட்சி ஆதாரம்"],
  "STEP 01 / IMAGE ACQUISITION":["चरण 01 / छवि प्राप्ति","படி 01 / படம் பெறுதல்"],
  "Bring the fundus image into RetinaX.":["फंडस छवि RetinaX में जोड़ें।","விழித்திரைப் படத்தை RetinaX-இல் சேர்க்கவும்."],
  "Select a patient fundus image to begin the screening workflow.":["स्क्रीनिंग शुरू करने के लिए रोगी की फंडस छवि चुनें।","பரிசோதனையைத் தொடங்க நோயாளியின் விழித்திரைப் படத்தைத் தேர்ந்தெடுக்கவும்."],
  "Upload fundus image":["फंडस छवि अपलोड करें","விழித்திரைப் படத்தைப் பதிவேற்றவும்"],
  "PNG, JPG or JPEG":["PNG, JPG या JPEG","PNG, JPG அல்லது JPEG"],
  "Choose image":["छवि चुनें","படத்தைத் தேர்ந்தெடு"],
  "No image selected":["कोई छवि नहीं चुनी गई","படம் தேர்ந்தெடுக்கப்படவில்லை"],
  "Analyze image":["छवि का विश्लेषण करें","படத்தைப் பகுப்பாய்வு செய்"],
  "IMAGE PREVIEW":["छवि पूर्वावलोकन","பட முன்னோட்டம்"],
  "Your fundus image will appear here":["आपकी फंडस छवि यहाँ दिखेगी","உங்கள் விழித்திரைப் படம் இங்கே தோன்றும்"],
  "STEP 02 / IMAGE QUALITY & INPUT SAFETY":["चरण 02 / छवि गुणवत्ता और सुरक्षा","படி 02 / படத் தரம் மற்றும் பாதுகாப்பு"],
  "Checking whether the image is safe to analyze.":["विश्लेषण के लिए छवि की उपयुक्तता जाँची जा रही है।","பகுப்பாய்வுக்கு படம் ஏற்றதா எனச் சரிபார்க்கப்படுகிறது."],
  "Focus, illumination, retinal visibility and gradability are verified first.":["पहले फोकस, रोशनी, रेटिना की दृश्यता और गुणवत्ता जाँची जाती है।","முதலில் தெளிவு, ஒளி, விழித்திரைத் தெரிவு மற்றும் தரம் சரிபார்க்கப்படும்."],
  "ANALYZING INPUT":["इनपुट विश्लेषण जारी","உள்ளீடு பகுப்பாய்வு"],
  "IMAGE ACCEPTED":["छवि स्वीकार की गई","படம் ஏற்கப்பட்டது"],
  "Gradable input":["जाँच योग्य छवि","மதிப்பிடத்தக்க படம்"],
  "Focus / sharpness":["फोकस / स्पष्टता","குவியம் / தெளிவு"],
  "Illumination":["प्रकाश","ஒளியமைப்பு"],
  "Retinal visibility":["रेटिना की दृश्यता","விழித்திரைத் தெரிவு"],
  "Gradability":["मूल्यांकन योग्यता","மதிப்பிடும் தகுதி"],
  "FUNDUS-DOMAIN GATEWAY":["फंडस सुरक्षा जाँच","விழித்திரைப் பாதுகாப்புச் சோதனை"],
  "Waiting for MATLAB safety checks":["MATLAB सुरक्षा जाँच की प्रतीक्षा","MATLAB பாதுகாப்புச் சோதனைக்காகக் காத்திருக்கிறது"],
  "STEP 03 / PREPROCESSING":["चरण 03 / पूर्व-प्रसंस्करण","படி 03 / முன்செயலாக்கம்"],
  "Original fundus vs. CLAHE-enhanced image.":["मूल फंडस बनाम CLAHE-संवर्धित छवि।","அசல் விழித்திரைப் படம் மற்றும் CLAHE மேம்படுத்திய படம்."],
  "ORIGINAL FUNDUS":["मूल फंडस","அசல் விழித்திரைப் படம்"],
  "CLAHE PREPROCESSED":["CLAHE संसाधित","CLAHE முன்செயலாக்கம்"],
  "Original prediction":["मूल अनुमान","அசல் கணிப்பு"],
  "CLAHE prediction":["CLAHE अनुमान","CLAHE கணிப்பு"],
  "Original confidence":["मूल विश्वास स्कोर","அசல் நம்பிக்கை மதிப்பு"],
  "CLAHE confidence":["CLAHE विश्वास स्कोर","CLAHE நம்பிக்கை மதிப்பு"],
  "STEP 04 / AI ANALYSIS":["चरण 04 / AI विश्लेषण","படி 04 / AI பகுப்பாய்வு"],
  "Multiple evidence streams, one screening workflow.":["कई साक्ष्य, एक स्क्रीनिंग प्रक्रिया।","பல ஆதாரங்கள், ஒரே பரிசோதனை நடைமுறை."],
  "Preprocessing":["पूर्व-प्रसंस्करण","முன்செயலாக்கம்"],
  "DR classification":["DR वर्गीकरण","DR வகைப்பாடு"],
  "Vessel analysis":["रक्तवाहिका विश्लेषण","இரத்த நாளப் பகுப்பாய்வு"],
  "Lesion evidence":["घाव के साक्ष्य","புண் ஆதாரம்"],
  "DR Grade":["DR स्तर","DR நிலை"],
  "Confidence":["विश्वास स्कोर","நம்பிக்கை மதிப்பு"],
  "Referable DR":["रेफरल योग्य DR","பரிந்துரை தேவைப்படும் DR"],
  "Review status":["समीक्षा स्थिति","மதிப்பாய்வு நிலை"],
  "STEP 05 / EXPLAINABILITY & EVIDENCE":["चरण 05 / व्याख्या और साक्ष्य","படி 05 / விளக்கம் மற்றும் ஆதாரம்"],
  "See what the model is looking at.":["देखें मॉडल किन क्षेत्रों पर ध्यान देता है।","மாதிரி கவனிக்கும் பகுதிகளைக் காணுங்கள்."],
  "GRAD-CAM OVERLAY":["Grad-CAM ओवरले","Grad-CAM மேலடுக்கு"],
  "VESSEL OVERLAY":["रक्तवाहिका ओवरले","இரத்த நாள மேலடுக்கு"],
  "LESION OVERLAY":["घाव ओवरले","புண் மேலடுக்கு"],
  "COMBINED STRUCTURAL EVIDENCE":["संयुक्त संरचनात्मक साक्ष्य","ஒருங்கிணைந்த கட்டமைப்பு ஆதாரம்"],
  "RETINAL LANDMARKS":["रेटिनल लैंडमार्क","விழித்திரை அடையாளப் புள்ளிகள்"],
  "EXPERIMENTAL":["प्रायोगिक","சோதனை நிலை"],
  "Estimated optic disc and experimental fovea candidate. Not clinically validated.":["अनुमानित ऑप्टिक डिस्क और प्रायोगिक फोविया स्थान। चिकित्सकीय रूप से सत्यापित नहीं।","மதிப்பிடப்பட்ட பார்வை வட்டு மற்றும் சோதனை ஃபோவியா இடம். மருத்துவ ரீதியாகச் சரிபார்க்கப்படவில்லை."],
  "Vessel area":["रक्तवाहिका क्षेत्र","இரத்த நாளப் பரப்பு"],
  "Total lesion area":["कुल घाव क्षेत्र","மொத்த புண் பரப்பு"],
  "CLAHE stability":["CLAHE स्थिरता","CLAHE நிலைத்தன்மை"],
  "Prediction safety":["अनुमान सुरक्षा","கணிப்புப் பாதுகாப்பு"],
  "STEP 06 / HUMAN-IN-THE-LOOP":["चरण 06 / मानवीय समीक्षा","படி 06 / மனித மதிப்பாய்வு"],
  "AI assists. A human makes the final review decision.":["AI सहायता करता है। अंतिम निर्णय मानव समीक्षक का है।","AI உதவுகிறது. இறுதி முடிவை மனித மதிப்பாய்வாளர் எடுக்கிறார்."],
  "AI SCREENING RESULT":["AI स्क्रीनिंग परिणाम","AI பரிசோதனை முடிவு"],
  "Model confidence":["मॉडल विश्वास स्कोर","மாதிரி நம்பிக்கை மதிப்பு"],
  "Priority human review":["प्राथमिक मानवीय समीक्षा","முன்னுரிமை மனித மதிப்பாய்வு"],
  "Accept AI result":["AI परिणाम स्वीकार करें","AI முடிவை ஏற்கவும்"],
  "Override / modify":["बदलें / संशोधित करें","மாற்றவும் / திருத்தவும்"],
  "Request recapture":["दोबारा छवि लें","மீண்டும் படம் எடுக்கவும்"],
  "Awaiting ophthalmologist review":["नेत्र विशेषज्ञ की समीक्षा लंबित","கண் மருத்துவர் மதிப்பாய்வு நிலுவையில்"],
  "STEP 07 / FINAL REPORT":["चरण 07 / अंतिम रिपोर्ट","படி 07 / இறுதி அறிக்கை"],
  "RetinaX screening report.":["RetinaX स्क्रीनिंग रिपोर्ट।","RetinaX பரிசோதனை அறிக்கை."],
  "SCREENING COMPLETE":["स्क्रीनिंग पूरी हुई","பரிசோதனை முடிந்தது"],
  "DR grade":["DR स्तर","DR நிலை"],
  "Screening status":["स्क्रीनिंग स्थिति","பரிசோதனை நிலை"],
  "Safety gateway":["सुरक्षा जाँच","பாதுகாப்புச் சோதனை"],
  "RECOMMENDATION":["अनुशंसा","பரிந்துரை"],
  "Print / Save Report":["रिपोर्ट प्रिंट / सेव करें","அறிக்கையை அச்சிடு / சேமி"],
  "Start new screening":["नई स्क्रीनिंग शुरू करें","புதிய பரிசோதனையைத் தொடங்கு"],
  "Back":["पीछे","பின்"],
  "Continue":["आगे बढ़ें","தொடரவும்"],
  "No image selected":["कोई छवि नहीं चुनी गई","படம் தேர்ந்தெடுக்கப்படவில்லை"]
};
const originalUIText = new WeakMap();
const dynamicResultIds = new Set([
 "fileName","qualityState","qualitySub","focusMetric","illumMetric","visibilityMetric","gradabilityMetric",
 "gatewayTitle","gatewaySub","gatewayScore","origPred","clahePred","origConf","claheConf",
 "preprocessStability","preprocessStabilityReason","severityState","resultGrade","resultConfidence",
 "resultReferable","resultReview","vesselArea","lesionArea","claheStability","predictionSafety",
 "reviewGrade","reviewConfidence","reviewReferral","reviewStatus","finalQuality","finalGrade",
 "finalConfidence","finalReferral","finalReview","finalRecommendation"
]);
function localizeUI(lang){
 document.documentElement.lang=lang;
 const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
 let node;
 while((node=walker.nextNode())){
  if(node.parentElement?.closest('script,style,select') || node.parentElement?.closest('[id]')?.id && dynamicResultIds.has(node.parentElement.closest('[id]').id))continue;
  if(!originalUIText.has(node))originalUIText.set(node,node.nodeValue);
  const original=originalUIText.get(node), key=original.trim();
  if(!key)continue;
  const translated=lang==='en'?key:(UI_TRANSLATIONS[key]?.[lang==='hi'?0:1]||key);
  node.nodeValue=original.replace(key,translated);
 }
 if(!selectedFile)$('fileName').textContent=lang==='en'?'No image selected':UI_TRANSLATIONS['No image selected'][lang==='hi'?0:1];
 $('languageSelect').value=lang;
 try{localStorage.setItem('retinax_language',lang);}catch(e){}
}
$('languageSelect').addEventListener('change',e=>localizeUI(e.target.value));
function setTheme(theme){
 document.documentElement.dataset.theme=theme;
 $('themeToggle').textContent=theme==='dark'?'☀':'☾';
 $('themeToggle').setAttribute('aria-label',theme==='dark'?'Switch to light mode':'Switch to dark mode');
 try{localStorage.setItem('retinax_theme',theme);}catch(e){}
}
$('themeToggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
let savedTheme='light',savedLang='en';
try{savedTheme=localStorage.getItem('retinax_theme')||'light';savedLang=localStorage.getItem('retinax_language')||'en';}catch(e){}
setTheme(savedTheme==='dark'?'dark':'light');
localizeUI(['en','hi','ta'].includes(savedLang)?savedLang:'en');
