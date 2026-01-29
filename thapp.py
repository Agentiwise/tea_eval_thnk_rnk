"""
Teacher Evaluation Workflow
Converts n8n workflow to Python script

This script:
1. Transcribes classroom audio using AssemblyAI
2. Parses curriculum PDF using LlamaIndex Cloud
3. Extracts training activities from curriculum using OpenAI
4. Evaluates teacher performance for each activity using OpenAI (parallel processing)
"""

import asyncio
import aiohttp
import re
import json
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
import streamlit as st


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """API Keys and Configuration"""
    ASSEMBLYAI_API_KEY: str = st.secrets["ASSEMBLYAI_API_KEY"]
    LLAMAINDEX_API_KEY: str = st.secrets["LLAMAINDEX_API_KEY"]
    GOOGLE_API_KEY: str = st.secrets["GOOGLE_API_KEY"]
    OPENAI_API_KEY: str = st.secrets["OPENAI_API_KEY"]
    
    # Processing settings
    TRANSCRIPTION_POLL_INTERVAL: int = 20  # seconds
    PARSING_POLL_INTERVAL: int = 5  # seconds
    EVALUATION_BATCH_SIZE: int = 25
    EVALUATION_BATCH_DELAY: float = 3.0  # seconds


@dataclass
class InputData:
    """Input data structure matching webhook body"""
    audio_file_drive_url: str
    curriculum_drive_url: str
    num_students: int
    custom_instruction: str
    class_name: str
    api_key: Optional[str] = None


# ============================================================================
# URL CONVERSION UTILITIES
# ============================================================================

def convert_drive_url_to_download(drive_url: str, use_google_api: bool = False, google_api_key: str = None) -> tuple[str, str]:
    """
    Convert Google Drive share URL to direct download URL.
    
    Args:
        drive_url: Google Drive share URL (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
        use_google_api: If True, uses Google Drive API format; otherwise uses simple export format
        google_api_key: Required if use_google_api is True
    
    Returns:
        Tuple of (download_url, file_id)
    """
    if not drive_url or 'drive.google.com/file/d/' not in drive_url:
        raise ValueError('Input URL is missing or not a valid Google Drive file URL.')
    
    try:
        # Extract file ID from URL
        id_part = drive_url.split('/d/')[1]
        file_id = id_part.split('/')[0]
        
        if use_google_api:
            # Google Drive API format (for curriculum/PDF)
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={google_api_key}"
        else:
            # Simple export format (for audio)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return download_url, file_id
    
    except IndexError:
        raise ValueError('Invalid Google Drive URL format.')


# ============================================================================
# ASSEMBLYAI TRANSCRIPTION
# ============================================================================

async def transcribe_audio(audio_url: str, config: Config) -> str:
    """
    Transcribe audio using AssemblyAI.
    
    Args:
        audio_url: Direct download URL for the audio file
        config: Configuration object with API keys
    
    Returns:
        Transcribed text
    """
    headers = {
        "Authorization": config.ASSEMBLYAI_API_KEY,
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Submit transcription request
        print("📤 Submitting audio for transcription...")
        async with session.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={"audio_url": audio_url}
        ) as response:
            result = await response.json()
            transcript_id = result["id"]
            print(f"   Transcript ID: {transcript_id}")
        
        # Step 2: Poll for completion
        print(f"⏳ Waiting for transcription (polling every {config.TRANSCRIPTION_POLL_INTERVAL}s)...")
        while True:
            await asyncio.sleep(config.TRANSCRIPTION_POLL_INTERVAL)
            
            async with session.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                headers=headers
            ) as response:
                result = await response.json()
                status = result.get("status")
                
                if status == "completed":
                    print("✅ Transcription completed!")
                    return result["text"]
                elif status == "error":
                    raise Exception(f"Transcription failed: {result.get('error')}")
                else:
                    print(f"   Status: {status}...")


# ============================================================================
# LLAMAINDEX PDF PARSING
# ============================================================================

async def parse_curriculum_pdf(curriculum_url: str, config: Config) -> str:
    """
    Parse curriculum PDF using LlamaIndex Cloud.
    
    Args:
        curriculum_url: Direct download URL for the curriculum PDF
        config: Configuration object with API keys
    
    Returns:
        Parsed markdown content
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {config.LLAMAINDEX_API_KEY}"
    }
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Upload for parsing
        print("📤 Uploading curriculum for parsing...")
        form_data = aiohttp.FormData()
        form_data.add_field("input_url", curriculum_url)
        
        async with session.post(
            "https://api.cloud.llamaindex.ai/api/v1/parsing/upload",
            headers=headers,
            data=form_data
        ) as response:
            result = await response.json()
            job_id = result["id"]
            print(f"   Job ID: {job_id}")
        
        # Step 2: Poll for completion
        print(f"⏳ Waiting for parsing (polling every {config.PARSING_POLL_INTERVAL}s)...")
        while True:
            await asyncio.sleep(config.PARSING_POLL_INTERVAL)
            
            async with session.get(
                f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}",
                headers=headers
            ) as response:
                result = await response.json()
                status = result.get("status")
                
                if status == "SUCCESS":
                    print("✅ Parsing completed!")
                    break
                elif status in ["FAILED", "ERROR"]:
                    raise Exception(f"Parsing failed: {result}")
                else:
                    print(f"   Status: {status}...")
        
        # Step 3: Get markdown result
        async with session.get(
            f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/markdown",
            headers=headers
        ) as response:
            result = await response.json()
            return result["markdown"]


# ============================================================================
# OPENAI ACTIVITY EXTRACTION
# ============================================================================

# JSON Schema for activity extraction
ACTIVITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "training_modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "activities": {"type": "string"},
                    "activity_brief_context": {"type": "string"}
                },
                "required": ["activities", "activity_brief_context"],
                "additionalProperties": False
            }
        }
    },
    "required": ["training_modules"],
    "additionalProperties": False
}

ACTIVITY_EXTRACTION_SYSTEM_PROMPT = """You are a strict Information Extraction Engine. Your sole purpose is to identify and consolidate unique, high-level learning activities from educational documents.

The Golden Rule: One Activity/Category, One Entry You must collapse all variations, levels, weeks, and specific instances into a single Parent Activity. If several entries share the same core methodology or category, you must merge them.

1. Extraction Logic (Strict Consolidation)

No Sub-parts or Levels: If the document lists "Auditory Memory: Bell" and "Auditory Memory: Drum," you must extract only "Auditory Memory."

No Sensorial Breakdowns: If the document lists "Sensorial: Weight," "Sensorial: Texture," and "Sensorial: Smell," you must extract only "Sensorial Activities."

No Progression: Ignore words like "Advanced," "Part 2," "Level B," or "Week 4."

Strip Suffixes: Remove anything following a hyphen (-), colon (:), or parenthesis () if it describes a specific variation, prop, or theme.

2. Qualification Criteria Extract only the Instructional Action.

Include: Rituals, drills, songs, games, physical movements, guided practices.

Exclude: Learning objectives, teacher "how-to" tips, materials lists, assessment scores, and metadata (headings/dates).

3. Output Format For each unique Parent Activity, provide:

Activity Name: The high-level, generic name only.

Brief Context: A one-sentence summary of the core action that covers all its variations.

Strict Negative Constraints:

NO DUPLICATES: If the activity appears in Week 1 and Week 20, or for different age groups, or with variations list it once.

NO VARIATIONS: Do not list "Language Song (Spanish)" and "Language Song (French)" separately. List "Language Song."

NO HIERARCHY: Do not use sub-bullets. Just a list of unique parent activities, with basic context or rxplanation of it.  """


def extract_activities(curriculum_markdown: str, openai_client: OpenAI) -> list[dict]:
    """
    Extract training activities from curriculum using OpenAI.
    
    Args:
        curriculum_markdown: Parsed curriculum in markdown format
        openai_client: OpenAI client instance
    
    Returns:
        List of activities with their context
    """
    print("🔍 Extracting activities from curriculum...")
    
    user_content = f"curriculum: \n\n{curriculum_markdown}"
    
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": ACTIVITY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "activity_extraction",
                "strict": True,
                "schema": ACTIVITY_EXTRACTION_SCHEMA
            }
        },
        timeout=600
    )
    
    result = json.loads(response.choices[0].message.content)
    activities = result.get("training_modules", [])
    
    print(f"✅ Extracted {len(activities)} activities")
    return activities


# ============================================================================
# OPENAI TEACHER EVALUATION
# ============================================================================

# JSON Schema for teacher evaluation
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "activity_name": {
            "type": "string",
            "description": "Name of the activity being evaluated"
        },
        "activity_context": {
            "type": ["string", "null"],
            "description": "Additional context about the activity if provided"
        },
        "status": {
            "type": "string",
            "enum": ["COMPLETED", "MISSING", "PARTIAL", "DEGRADED_QUALITY"],
            "description": "Status of the activity in the transcript"
        },
        "clarity_assessment": {
            "type": "string",
            "description": "5-7 line objective assessment of clarity of instruction and tone"
        },
        "clarity_rating": {
            "type": "integer",
            "description": "Rating for clarity (1-10, or 0 if missing)"
        },
        "interaction_assessment": {
            "type": "string",
            "description": "5-7 line objective assessment of student interaction and verification"
        },
        "interaction_rating": {
            "type": "integer",
            "description": "Rating for student interaction (1-10, or 0 if missing)"
        },
        "pacing_assessment": {
            "type": "string",
            "description": "5-7 line objective assessment of pacing and flow"
        },
        "pacing_rating": {
            "type": "integer",
            "description": "Rating for pacing and flow (1-10, or 0 if missing)"
        },
        "overall_rating": {
            "type": "number",
            "description": "Average of three category ratings, rounded to 1 decimal place"
        },
        "comments": {
            "type": "string",
            "description": "5-7 line holistic feedback and final assessment"
        }
    },
    "required": [
        "activity_name", "activity_context", "status",
        "clarity_assessment", "clarity_rating",
        "interaction_assessment", "interaction_rating",
        "pacing_assessment", "pacing_rating",
        "overall_rating", "comments"
    ],
    "additionalProperties": False
}

EVALUATION_SYSTEM_PROMPT = """## ROLE You are an objective educational assessor specializing in early childhood education (toddlers and small children aged 1-4 years). Your task is to evaluate teacher performance based on classroom audio transcripts.  ## INPUT PROVIDED You will receive: 1. **Transcript**: A 30-60 minute classroom audio transcript with speaker labels 2. **Activity Name**: The specific classroom activity to evaluate (e.g., "WELCOME CIRCLE & AFFIRMATIONS", "STORY WITH PROPS") 3. **Activity Context** (optional): Brief description of the activity's purpose, target age group, or learning objectives  ## EVALUATION FRAMEWORK  Assess the teacher across THREE equally-weighted categories:  ### 1. CLARITY OF INSTRUCTION & TONE Evaluate: - Content delivery (clear, age-appropriate language) - Quality of explanations (simple, concrete, understandable) - Use of examples (relevant, engaging, relatable) - Voice tone and volume (warm, encouraging, audible)  ### 2. STUDENT INTERACTION & VERIFICATION Evaluate: - Inclusivity (engaging all children, addressing individual needs) - Questioning techniques (open-ended, age-appropriate, encouraging participation) - Response to student input (acknowledging, validating, building upon) - Checking for understanding (appropriate verification methods)  ### 3. PACING & FLOW Evaluate: - Smooth transitions (logical progression, minimal disruption) - Alignment with curriculum/activity objectives (staying on topic) - Time management (appropriate duration, no rushing or dragging) - Energy and engagement maintenance  ## RATING SCALE (1-10)  Apply this scale objectively: - **10-9**: Exceptional - Masterful execution, age-perfect, highly engaging - **8-7**: Strong - Effective delivery, good engagement, minor areas for polish - **6-5**: Adequate - Meets basic standards, some inconsistencies, room for improvement - **4-3**: Needs Improvement - Multiple gaps, unclear delivery, or pacing issues - **2-1**: Unsatisfactory - Significant deficiencies, inappropriate for age group, poor execution  ## RATING METHODOLOGY 1. Assign individual ratings (1-10) for each of the three categories 2. Calculate overall rating as: (Clarity Rating + Interaction Rating + Pacing Rating) / 3 3. Round overall rating to one decimal place  ## SPECIAL CASES  ### If Activity is Missing from Transcript: - Set all ratings to 0 - Mark status as "MISSING" - In assessments, state: "Activity not found in provided transcript" - In comments, note: "Unable to evaluate - activity not conducted or not captured in recording"  ### If Activity is Partially Present: - Evaluate what is available - Note the limitation in each assessment - Mark status as "PARTIAL" - Reduce ratings proportionally based on available content  
### The evaluation should be really objective such that same input un twice gives same results. Try to keep the evaluation lenient and mild. the transcription is majorly for the teacher's audio, so ignore the student's response. te clas can be individual or group(4-5), so dont worry about the teacher just interactiving with one or two children
### If Transcript Quality is Poor: - Note specific issues (inaudible sections, unclear speakers) - Evaluate based on audible portions - Mark status as "DEGRADED_QUALITY" - Indicate limitations in comments  ## OUTPUT REQUIREMENTS  Provide assessment in exactly 5-7 lines for each category: - **Lines 1-3**: Objective observations of what occurred - **Lines 4-5**: Assessment against evaluation criteria - **Lines 6-7**: Specific strengths or areas needing attention  **Comments Section**: Provide holistic 5-7 line feedback including: - Overall performance summary - Key strengths demonstrated - Primary areas for development - Contextual considerations (if any) - Final evaluative statement  ## WRITING STYLE - Objective and evidence-based - Professional and neutral tone - Specific and concrete (reference transcript moments) - Age-appropriate expectations (consider toddler behavior) - No subjective judgments or assumptions - Quantify observations where possible (e.g., "called on 4 different students")    ## CRITICAL INSTRUCTIONS 1. Analyze ONLY the specified activity from the transcript 2. Base ratings on observable evidence, not assumptions 3. Consider age-appropriateness (toddler/small children standards) 4. Maintain consistency in rating application across all assessments 5. Ensure each assessment section is exactly 5-7 lines 6. Return ONLY the JSON object, no additional text 7. Use proper JSON formatting with escaped characters where needed 8. All ratings must be integers between 1-10 9. Overall rating must be calculated as average of three category ratings  ---"""


def evaluate_single_activity(
    activity: dict,
    transcript: str,
    cust_inst: str,
    n_studs: str,
    openai_client: OpenAI
) -> dict:
    """
    Evaluate teacher performance for a single activity.
    
    Args:
        activity: Activity dict with 'activities' and 'activity_brief_context'
        transcript: Full class transcript
        openai_client: OpenAI client instance
    
    Returns:
        Evaluation result dict
    """
    user_content = f"""Activity to search and assess for:
{activity['activities']}

{activity['activity_brief_context']}

number of students in the class: {n_studs}. Student responses will not be tresent in the transcript so avoid rating on the basis of that.
any optional instructions to focus: {cust_inst}

------------------------------------------------------------
complete class transcript
{transcript}"""

    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.4,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "teacher_evaluation",
                "strict": True,
                "schema": EVALUATION_SCHEMA
            }
        },
        timeout=600
    )
    
    return json.loads(response.choices[0].message.content)


async def evaluate_activities_parallel(
    activities: list[dict],
    transcript: str,
    cust_inst: str,
    n_studs: str,
    openai_client: OpenAI,
    config: Config
) -> list[dict]:
    """
    Evaluate all activities in parallel with batching.
    
    Args:
        activities: List of activities to evaluate
        transcript: Full class transcript
        openai_client: OpenAI client instance
        config: Configuration with batch settings
    
    Returns:
        List of evaluation results
    """
    print(f"📊 Evaluating {len(activities)} activities (batch size: {config.EVALUATION_BATCH_SIZE})...")
    
    results = []
    
    # Process in batches
    for batch_start in range(0, len(activities), config.EVALUATION_BATCH_SIZE):
        batch_end = min(batch_start + config.EVALUATION_BATCH_SIZE, len(activities))
        batch = activities[batch_start:batch_end]
        batch_num = (batch_start // config.EVALUATION_BATCH_SIZE) + 1
        total_batches = (len(activities) + config.EVALUATION_BATCH_SIZE - 1) // config.EVALUATION_BATCH_SIZE
        
        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} activities)...")
        
        # Run batch in parallel using asyncio
        loop = asyncio.get_event_loop()
        batch_tasks = [
            loop.run_in_executor(
                None,
                evaluate_single_activity,
                activity,
                transcript,
                cust_inst,
                n_studs,
                openai_client
            )
            for activity in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"   ⚠️ Error evaluating activity {batch_start + i + 1}: {result}")
                results.append({
                    "activity_name": batch[i].get("activities", "Unknown"),
                    "activity_context": batch[i].get("activity_brief_context"),
                    "status": "ERROR",
                    "error": str(result),
                    "clarity_assessment": "Evaluation failed due to error",
                    "clarity_rating": 0,
                    "interaction_assessment": "Evaluation failed due to error",
                    "interaction_rating": 0,
                    "pacing_assessment": "Evaluation failed due to error",
                    "pacing_rating": 0,
                    "overall_rating": 0,
                    "comments": f"Error during evaluation: {result}"
                })
            else:
                results.append(result)
        
        # Delay between batches (except for last batch)
        if batch_end < len(activities):
            print(f"   Waiting {config.EVALUATION_BATCH_DELAY}s before next batch...")
            await asyncio.sleep(config.EVALUATION_BATCH_DELAY)
    
    print(f"✅ Completed {len(results)} evaluations")
    return results


# ============================================================================
# MAIN WORKFLOW FUNCTION
# ============================================================================

async def process_teacher_evaluation(input_data: InputData, config: Config = None) -> list[dict]:
    """
    Main workflow function - processes teacher evaluation from audio and curriculum.
    
    This is the entry point that replaces the webhook.
    
    Args:
        input_data: Input data matching webhook body structure
        config: Optional configuration (uses defaults if not provided)
    
    Returns:
        List of evaluation results for each activity
    """
    if config is None:
        config = Config()
    
    print("=" * 60)
    print("🎓 TEACHER EVALUATION WORKFLOW")
    print("=" * 60)
    print(f"Class: {input_data.class_name}")
    print(f"Students: {input_data.num_students}")
    print(f"Custom Instructions: {input_data.custom_instruction}")
    print("=" * 60)
    cust_inst=input_data.custom_instruction
    n_studs=input_data.num_students
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else OpenAI()
    
    # Step 1: Convert audio URL and transcribe
    print("\n📝 STAGE 1: AUDIO TRANSCRIPTION")
    print("-" * 40)
    audio_download_url, audio_file_id = convert_drive_url_to_download(
        input_data.audio_file_drive_url,
        use_google_api=False
    )
    print(f"Audio file ID: {audio_file_id}")
    
    transcript = await transcribe_audio(audio_download_url, config)
    print(f"Transcript length: {len(transcript)} characters")
    
    # Step 2: Parse curriculum PDF
    print("\n📚 STAGE 2: CURRICULUM PARSING")
    print("-" * 40)
    curriculum_download_url, curriculum_file_id = convert_drive_url_to_download(
        input_data.curriculum_drive_url,
        use_google_api=True,
        google_api_key=config.GOOGLE_API_KEY
    )
    print(f"Curriculum file ID: {curriculum_file_id}")
    
    curriculum_markdown = await parse_curriculum_pdf(curriculum_download_url, config)
    print(f"Curriculum length: {len(curriculum_markdown)} characters")
    
    # Step 3: Extract activities
    print("\n🎯 STAGE 3: ACTIVITY EXTRACTION")
    print("-" * 40)
    activities = extract_activities(curriculum_markdown, openai_client)
    
    for i, activity in enumerate(activities, 1):
        print(f"   {i}. {activity['activities'][:50]}...")
    
    # Step 4: Evaluate each activity
    print("\n📊 STAGE 4: TEACHER EVALUATION")
    print("-" * 40)
    evaluations = await evaluate_activities_parallel(
        activities,
        transcript,
        cust_inst,
        n_studs,
        openai_client,
        config
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 EVALUATION SUMMARY")
    print("=" * 60)
    
    for eval_result in evaluations:
        status_emoji = {
            "COMPLETED": "✅",
            "MISSING": "❌",
            "PARTIAL": "⚠️",
            "DEGRADED_QUALITY": "📉",
            "ERROR": "❗"
        }.get(eval_result.get("status"), "❓")
        
        print(f"{status_emoji} {eval_result['activity_name']}: {eval_result['overall_rating']}/10")
    
    return evaluations


# ============================================================================
# SYNCHRONOUS WRAPPER
# ============================================================================

def run_teacher_evaluation(
    audio_file_drive_url: str,
    curriculum_drive_url: str,
    num_students: int = 1,
    custom_instruction: str = "",
    class_name: str = "Unnamed Class",
    openai_api_key: str = None,
    **kwargs
) -> list[dict]:
    """
    Synchronous wrapper for the teacher evaluation workflow.
    
    Args:
        audio_file_drive_url: Google Drive URL for the classroom audio
        curriculum_drive_url: Google Drive URL for the curriculum PDF
        num_students: Number of students in the class
        custom_instruction: Custom evaluation instructions
        class_name: Name of the class
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        **kwargs: Additional configuration overrides
    
    Returns:
        List of evaluation results
    """
    input_data = InputData(
        audio_file_drive_url=audio_file_drive_url,
        curriculum_drive_url=curriculum_drive_url,
        num_students=num_students,
        custom_instruction=custom_instruction,
        class_name=class_name
    )
    
    config = Config()
    if openai_api_key:
        config.OPENAI_API_KEY = openai_api_key
    
    # Apply any additional config overrides
    for key, value in kwargs.items():
        if hasattr(config, key.upper()):
            setattr(config, key.upper(), value)
    
    return asyncio.run(process_teacher_evaluation(input_data, config))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage matching the pinned webhook data
    results = run_teacher_evaluation(
        audio_file_drive_url="https://drive.google.com/file/d/1GqjOPPn72MWyWXWIxMsoTCVTVMeXWuuw/view?usp=drive_link",
        curriculum_drive_url="https://drive.google.com/file/d/16uXFJ14vWddYIPP0U8WZefvyNtHIKD5Y/view?usp=drive_link",
        num_students=1,
        custom_instruction="more lenient evaluation",
        class_name="class 1 28th jan delhi abc",
    )
    
    # Print full results as JSON
    print("\n" + "=" * 60)
    print("📄 FULL RESULTS (JSON)")
    print("=" * 60)
    print(json.dumps(results, indent=2))