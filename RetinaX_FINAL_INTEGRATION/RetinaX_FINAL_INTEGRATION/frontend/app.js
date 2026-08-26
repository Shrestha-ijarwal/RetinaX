
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
}
const num=(x,n)=>typeof x==="number"?x.toFixed(n):"—", pct01=x=>typeof x==="number"?(100*x).toFixed(2)+"%":"—";
$("acceptReview").onclick=async()=>{await saveReview("ACCEPTED_BY_HUMAN","Human reviewer accepted the AI screening result.");$("reviewStatus").textContent="✓ AI result accepted by human reviewer";$("reviewStatus").style.color="#15966b";$("finalReview").textContent="AI RESULT ACCEPTED";};
$("overrideReview").onclick=async()=>{const g=prompt("Enter clinician final grade / note:","Specialist review required");if(g){await saveReview("OVERRIDDEN_BY_HUMAN",g);$("reviewStatus").textContent="Reviewer override recorded: "+g;$("reviewStatus").style.color="#1769e0";$("finalReview").textContent="REVIEWER OVERRIDE";}};
$("recaptureReview").onclick=async()=>{await saveReview("RECAPTURE_REQUESTED","Acquire a new gradable fundus image.");$("reviewStatus").textContent="Image recapture requested";$("reviewStatus").style.color="#d94b5d";$("finalReview").textContent="RECAPTURE REQUESTED";$("finalRecommendation").textContent="Acquire a new gradable fundus image before final interpretation.";};
async function saveReview(decision,note){if(!latestResult?.case_id)return;try{await fetch(`/api/review/${latestResult.case_id}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision,note})});}catch(e){console.error(e);}}
$("backBtn").onclick=()=>goTo(current-1);$("nextBtn").onclick=()=>goTo(current+1);
$("newCaseBtn").onclick=()=>location.reload();
goTo(0);
