"""
============================================================
SAP Generative AI Hub SDK — Full Interview Learning Guide
============================================================

IMPORTANT NOTE:
    'generative-ai-hub-sdk' is DEPRECATED as of 2024.
    The new official package is: sap-ai-sdk-gen

Install:
    pip install "sap-ai-sdk-gen[all]"
    pip install langchain-community langchain-text-splitters faiss-cpu

Configuration (~/.aicore/config.json or environment variables):
{
    "AICORE_AUTH_URL":         "https://<tenant>.authentication.sap.hana.ondemand.com/oauth/token",
    "AICORE_CLIENT_ID":        "your-client-id",
    "AICORE_CLIENT_SECRET":    "your-client-secret",
    "AICORE_BASE_URL":         "https://api.ai.core.prod.us20.hanacloud.ondemand.com",
    "AICORE_RESOURCE_GROUP":   "default",
    "AICORE_ORCHESTRATION_DEPLOYMENT_URL": "https://<your-orch-deployment-url>"
}

Architecture Summary:
    ┌──────────────────────────────────────────────────────┐
    │              SAP Generative AI Hub                   │
    │                                                      │
    │  User Input                                          │
    │     ↓                                                │
    │  [Prompt Template]  ← define dynamic placeholders   │
    │     ↓                                                │
    │  [Data Masking]     ← anonymize PII before LLM      │
    │     ↓                                                │
    │  [Grounding]        ← inject retrieved context      │
    │     ↓                                                │
    │  [LLM]              ← gpt-4o, claude, gemini etc.   │
    │     ↓                                                │
    │  [Content Filter]   ← safety check on output        │
    │     ↓                                                │
    │  Final Response                                      │
    └──────────────────────────────────────────────────────┘
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================

# OrchestrationConfig: the master config object that bundles
# template + llm + grounding + masking + filtering together
from gen_ai_hub.orchestration.models.config import OrchestrationConfig

# LLM: defines which model to use and its inference parameters
from gen_ai_hub.orchestration.models.llm import LLM

# Message types: used to construct role-based prompt messages
# SystemMessage: sets the AI's behavior/persona
# UserMessage:   the user's input message (supports {{?placeholder}} syntax)
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage

# Template: holds the prompt structure with message roles and placeholders
# TemplateValue: runtime value injected into a {{?placeholder}} at run time
from gen_ai_hub.orchestration.models.template import Template, TemplateValue

# OrchestrationService: the main client that connects to your SAP AI Core
# orchestration deployment and runs/streams the pipeline
from gen_ai_hub.orchestration.service import OrchestrationService

# Document grounding: enables Retrieval-Augmented Generation (RAG)
# DocumentGrounding:       config for grounding module
# DocumentGroundingFilter: defines WHICH data source to search
# GroundingModule:         wraps grounding config into the pipeline
# GroundingType:           enum — currently DOCUMENT_GROUNDING_SERVICE
# DataRepositoryType:      enum — VECTOR (custom) or help.sap.com
from gen_ai_hub.orchestration.models.document_grounding import (
    DocumentGrounding,
    DocumentGroundingFilter,
    GroundingModule,
    GroundingType,
    DataRepositoryType,
)

# Content filtering: uses Azure Content Safety to block harmful outputs
# AzureContentFilter:  defines thresholds (0=strictest, 6=most permissive)
# ContentFilterConfig: wraps input and output filter settings
from gen_ai_hub.orchestration.models.content_filter import (
    AzureContentFilter,
    ContentFilterConfig,
)

# Data masking: PII protection via SAP Data Privacy Integration
# SAPDataPrivacyIntegration: provider for SAP's DPI service
# MaskingMethod: ANONYMIZATION (irreversible) or PSEUDONYMIZATION (reversible)
# ProfileEntity: types of PII to detect (EMAIL, PHONE, PERSON, etc.)
from gen_ai_hub.orchestration.models.sap_data_privacy_integration import (
    SAPDataPrivacyIntegration,
    MaskingMethod,
    ProfileEntity,
)

# DataMasking: wraps one or more masking providers into the pipeline
from gen_ai_hub.orchestration.models.data_masking import DataMasking

# List type hint for conversation history management
from typing import List


# ============================================================
# SECTION 2: INITIALIZE OrchestrationService
# ============================================================

def create_orchestration_service(deployment_url: str) -> OrchestrationService:
    """
    Creates and returns an OrchestrationService instance.

    What is OrchestrationService?
        - The main entry point to talk to SAP AI Core Orchestration
        - It wraps your HTTP calls (auth token injection, headers, etc.)
        - You call .run() for synchronous or .stream() for streaming responses

    Args:
        deployment_url: The URL of your deployed Orchestration service
                        (found in SAP AI Launchpad → ML Operations → Deployments)

    Returns:
        An initialized OrchestrationService ready to execute pipelines
    """
    # OrchestrationService reads credentials automatically from:
    # ~/.aicore/config.json OR environment variables
    # You only need to provide the specific deployment URL
    service = OrchestrationService(api_url=deployment_url)
    return service


# ============================================================
# SECTION 3: DEFINE LLM
# ============================================================

def create_llm(model_name: str = "gpt-4o") -> LLM:
    """
    Defines which LLM to use and its inference parameters.

    What is LLM here?
        - NOT the model itself, just a configuration descriptor
        - SAP AI Core resolves the model name to an actual deployment
        - Supports: gpt-4o, gpt-4o-mini, gemini-1.5-pro, claude-3.5-sonnet, etc.

    Parameters explained:
        name:        model name exactly as configured in AI Core
        version:     "latest" or a specific version string
        max_tokens:  maximum tokens in the LLM response
        temperature: 0.0 = deterministic/factual, 1.0 = creative/random
        top_p:       nucleus sampling threshold (alternative to temperature)

    Returns:
        LLM config object
    """
    llm = LLM(
        name=model_name,          # model name registered in your AI Core
        version="latest",         # always use latest unless pinning
        parameters={
            "max_tokens": 1024,   # cap on response length
            "temperature": 0.3,   # low = more factual answers
            "top_p": 0.95,        # consider top 95% probable tokens
        }
    )
    return llm


# ============================================================
# SECTION 4: DEFINE PROMPT TEMPLATE
# ============================================================

def create_template(system_prompt: str, user_prompt_with_placeholders: str) -> Template:
    """
    Defines the prompt structure with dynamic placeholders.

    What is Template?
        - Defines the role-based messages sent to the LLM
        - Supports {{?variable_name}} placeholder syntax
        - Placeholders are filled at runtime via TemplateValue objects
        - Follows the standard chat format: system → user

    Placeholder syntax:
        {{?placeholder_name}}  ← double curly brace + ? + name
        Example: "Answer this: {{?user_query}}"

    Built-in grounding placeholder:
        {{?grounding_response}}  ← automatically filled by the grounding module
        You must include this in your UserMessage when using grounding

    Args:
        system_prompt:                    The system instruction string
        user_prompt_with_placeholders:    User message with {{?var}} syntax

    Returns:
        Template object ready to be used in OrchestrationConfig
    """
    template = Template(
        messages=[
            # SystemMessage: sets the persona/behavior of the LLM
            # Always provide clear instructions here
            SystemMessage(system_prompt),

            # UserMessage: the actual user query
            # Use {{?var}} placeholders for dynamic content
            UserMessage(user_prompt_with_placeholders),
        ]
    )
    return template


# ============================================================
# SECTION 5: BASIC RUN (No Grounding, No Masking)
# ============================================================

def run_basic_completion(
    service: OrchestrationService,
    llm: LLM,
    question: str
) -> str:
    """
    Runs the simplest orchestration pipeline: just template + LLM.

    Use this when:
        - You just want a basic chat completion
        - No RAG, no PII masking, no content filtering needed

    Flow:
        User Input → Template → LLM → Response

    Args:
        service:  initialized OrchestrationService
        llm:      LLM config object
        question: the user's question string

    Returns:
        The LLM's response text
    """
    # Create a simple template with one placeholder: {{?question}}
    template = create_template(
        system_prompt="You are a helpful SAP expert.",
        user_prompt_with_placeholders="Answer the following: {{?question}}"
    )

    # OrchestrationConfig bundles template + llm
    # Think of this as the "blueprint" of your pipeline
    config = OrchestrationConfig(
        template=template,
        llm=llm,
    )

    # .run() executes the full pipeline synchronously
    # template_values: list of TemplateValue to fill placeholders
    result = service.run(
        config=config,
        template_values=[
            # TemplateValue maps placeholder name → runtime value
            TemplateValue(name="question", value=question)
        ]
    )

    # Navigate the response structure:
    # result.orchestration_result → the CompletionResult
    # .choices[0]                 → first (and usually only) choice
    # .message.content            → the actual text response
    response_text = result.orchestration_result.choices[0].message.content
    return response_text


# ============================================================
# SECTION 6: STREAMING
# ============================================================

def run_streaming(
    service: OrchestrationService,
    llm: LLM,
    question: str
) -> None:
    """
    Streams the LLM response token by token (chunk by chunk).

    When to use streaming?
        - Chat UIs where you want real-time token display
        - Long responses where waiting is frustrating
        - Improves perceived performance significantly

    How it works:
        Instead of waiting for the full response, the server
        sends partial chunks (deltas) as they are generated.
        Each chunk contains the next piece of text.
    """
    template = create_template(
        system_prompt="You are a helpful assistant.",
        user_prompt_with_placeholders="Explain: {{?topic}}"
    )
    config = OrchestrationConfig(template=template, llm=llm)

    # .stream() returns a generator instead of a full response
    # Each iteration gives you a chunk with partial content
    stream = service.stream(
        config=config,
        template_values=[TemplateValue(name="topic", value=question)]
    )

    print("Streaming response:")
    for chunk in stream:
        # Each chunk has orchestration_result with partial message
        # We print without newline to simulate real-time typing
        if chunk.orchestration_result:
            delta = chunk.orchestration_result.choices[0].delta.content or ""
            print(delta, end="", flush=True)
    print()  # newline at end


# ============================================================
# SECTION 7: GROUNDING — SAP Help Portal (help.sap.com)
# ============================================================

def run_with_sap_help_grounding(
    service: OrchestrationService,
    llm: LLM,
    user_query: str
) -> dict:
    """
    Uses SAP Help Portal as the grounding source (RAG).

    What is Grounding / RAG?
        Retrieval-Augmented Generation (RAG) = Retrieval + Generation
        Step 1 (Retrieval): Search a knowledge base for relevant chunks
        Step 2 (Generation): Pass retrieved chunks as context to LLM
        Result: LLM answers from real documents, not just training data

    Why use SAP Help Portal grounding?
        - SAP Help Portal has official SAP product documentation
        - Perfect for answering questions about SAP products accurately
        - No need to maintain your own vector store

    Template placeholders involved:
        {{?user_query}}         → filled by you (the search query)
        {{?grounding_response}} → AUTO-FILLED by the grounding module
                                  with relevant document chunks

    Returns:
        dict with 'answer' and 'retrieved_context'
    """
    # Template must include {{?grounding_response}} for context injection
    # The grounding module finds relevant docs and fills this placeholder
    template = Template(
        messages=[
            SystemMessage(
                "You are an SAP product expert. "
                "Answer ONLY based on the provided context."
            ),
            UserMessage(
                "Context from SAP Help Portal:\n"
                "{{ ?grounding_response }}\n\n"   # auto-filled by grounding
                "Question: {{ ?user_query }}"      # filled by you
            ),
        ]
    )

    # DocumentGroundingFilter: specifies WHERE to search
    # data_repository_type="help.sap.com" → searches SAP Help Portal
    filters = [
        DocumentGroundingFilter(
            id="sap_help",                          # arbitrary identifier
            data_repository_type="help.sap.com"    # built-in SAP source
        )
    ]

    # DocumentGrounding: specifies HOW to use the grounding
    # input_params:  which placeholder values drive the search query
    # output_param:  which placeholder receives the retrieved text
    grounding_config = GroundingModule(
        type=GroundingType.DOCUMENT_GROUNDING_SERVICE.value,
        config=DocumentGrounding(
            input_params=["user_query"],       # search using this value
            output_param="grounding_response", # inject results here
            filters=filters
        )
    )

    # Bundle everything into OrchestrationConfig
    config = OrchestrationConfig(
        template=template,
        llm=llm,
        grounding=grounding_config,   # attach grounding module
    )

    result = service.run(
        config=config,
        template_values=[
            TemplateValue(name="user_query", value=user_query)
        ]
    )

    # Main LLM answer
    answer = result.orchestration_result.choices[0].message.content

    # Retrieved context: what was actually fetched from the knowledge base
    # Useful for debugging and showing sources to end users
    retrieved_context = None
    if result.module_results and result.module_results.grounding:
        retrieved_context = result.module_results.grounding.data.get(
            "grounding_result"
        )

    return {
        "answer": answer,
        "retrieved_context": retrieved_context
    }


# ============================================================
# SECTION 8: GROUNDING — Custom Vector Store
# ============================================================

def run_with_vector_grounding(
    service: OrchestrationService,
    llm: LLM,
    user_query: str,
    vector_repository_id: str,
    max_chunks: int = 5
) -> dict:
    """
    Uses your own vector data repository for grounding (custom RAG).

    When to use this vs SAP Help Portal grounding?
        - Use SAP Help Portal: for official SAP product questions
        - Use Vector Store: for your OWN documents (HR policies, manuals, etc.)

    What is a vector data repository?
        - Your documents are chunked, embedded (converted to vectors)
        - Stored in SAP AI Core's Document Grounding Service
        - At query time, the user query is embedded and similar chunks are
          retrieved using cosine similarity
        - You set this up via SAP AI Launchpad → Document Grounding

    Args:
        vector_repository_id: The ID from SAP AI Launchpad for your vector store
        max_chunks:           How many document chunks to retrieve (default: 5)
                              More chunks = more context but higher token usage

    Returns:
        dict with 'answer' and 'retrieved_context'
    """
    template = Template(
        messages=[
            SystemMessage(
                "You are a helpful assistant. "
                "Answer ONLY based on the provided context. "
                "If the context doesn't contain the answer, say 'I don't know'."
            ),
            UserMessage(
                "Relevant documents:\n"
                "{{ ?grounding_response }}\n\n"
                "User question: {{ ?user_query }}"
            ),
        ]
    )

    # For custom vector stores, use DataRepositoryType.VECTOR
    # and provide the actual repository ID from AI Launchpad
    filters = [
        DocumentGroundingFilter(
            id="my_vector_store",
            data_repositories=[vector_repository_id],  # your repo ID
            data_repository_type=DataRepositoryType.VECTOR.value,
            search_config={
                "max_chunk_count": max_chunks  # control how many chunks to retrieve
            }
        )
    ]

    grounding_config = GroundingModule(
        type=GroundingType.DOCUMENT_GROUNDING_SERVICE.value,
        config=DocumentGrounding(
            input_params=["user_query"],
            output_param="grounding_response",
            filters=filters
        )
    )

    config = OrchestrationConfig(
        template=template,
        llm=llm,
        grounding=grounding_config,
    )

    result = service.run(
        config=config,
        template_values=[
            TemplateValue(name="user_query", value=user_query)
        ]
    )

    answer = result.orchestration_result.choices[0].message.content
    retrieved_context = None
    if result.module_results and result.module_results.grounding:
        retrieved_context = result.module_results.grounding.data.get(
            "grounding_result"
        )

    return {"answer": answer, "retrieved_context": retrieved_context}


# ============================================================
# SECTION 9: DATA MASKING (PII PROTECTION)
# ============================================================

def create_data_masking(method: str = "anonymization") -> DataMasking:
    """
    Creates a data masking configuration to protect PII before sending to LLM.

    What is Data Masking?
        - PII (Personally Identifiable Information) in user input is detected
          and replaced before the text reaches the LLM
        - Uses SAP Data Privacy Integration (DPI) service
        - Critical for enterprise compliance (GDPR, SOC2, etc.)

    Anonymization vs Pseudonymization:
        ANONYMIZATION:    "John Smith emailed john@sap.com"
                          → "[PERSON] emailed [EMAIL]"
                          Irreversible — cannot recover original values

        PSEUDONYMIZATION: "John Smith emailed john@sap.com"
                          → "ENTITY_001 emailed ENTITY_002"
                          Reversible — SDK can map tokens back to originals
                          Useful when you need original values in post-processing

    PII Types (ProfileEntity):
        EMAIL     → email addresses
        PHONE     → phone numbers
        PERSON    → people's names
        ADDRESS   → physical addresses
        SAP_IDS_INTERNAL → SAP-specific identifiers

    Args:
        method: "anonymization" or "pseudonymization"

    Returns:
        DataMasking config object
    """
    # Choose masking method based on requirement
    masking_method = (
        MaskingMethod.ANONYMIZATION
        if method == "anonymization"
        else MaskingMethod.PSEUDONYMIZATION
    )

    # SAPDataPrivacyIntegration: SAP's built-in PII detection engine
    dpi_provider = SAPDataPrivacyIntegration(
        method=masking_method,
        entities=[
            ProfileEntity.EMAIL,    # detect and mask emails
            ProfileEntity.PHONE,    # detect and mask phone numbers
            ProfileEntity.PERSON,   # detect and mask names
        ],
        mask_grounding_input=True   # also mask before sending to grounding/vector search
    )

    # DataMasking wraps one or more providers
    # SAP DPI is currently the supported provider
    data_masking = DataMasking(providers=[dpi_provider])
    return data_masking


# ============================================================
# SECTION 10: CONTENT FILTERING
# ============================================================

def create_content_filter(strictness: int = 0) -> ContentFilterConfig:
    """
    Creates Azure Content Safety filters for input and output.

    What is Content Filtering?
        - Automatically blocks harmful content in user inputs AND LLM outputs
        - Powered by Azure Content Safety
        - Applies to both directions: what goes IN and what comes OUT

    Threshold levels:
        0 → Block all (most strict)  ← recommended for enterprise
        2 → Allow low severity
        4 → Allow medium severity
        6 → Allow most content (least strict)

    Categories:
        hate      → hate speech and discrimination
        self_harm → self-harm encouragement
        sexual    → explicit sexual content
        violence  → violent content

    Args:
        strictness: 0 (strictest) to 6 (most permissive)

    Returns:
        ContentFilterConfig with both input and output filters set
    """
    # Create filter for user INPUT (before LLM)
    input_filter = AzureContentFilter(
        hate=strictness,
        self_harm=strictness,
        sexual=strictness,
        violence=strictness
    )

    # Create filter for LLM OUTPUT (before returning to user)
    output_filter = AzureContentFilter(
        hate=strictness,
        self_harm=strictness,
        sexual=strictness,
        violence=strictness
    )

    # ContentFilterConfig bundles both input and output filters
    return ContentFilterConfig(input=input_filter, output=output_filter)


# ============================================================
# SECTION 11: FULL ENTERPRISE PIPELINE
#             Grounding + Masking + Filtering combined
# ============================================================

def run_full_enterprise_pipeline(
    service: OrchestrationService,
    llm: LLM,
    user_query: str,
    vector_repository_id: str
) -> dict:
    """
    Full production-grade pipeline with ALL modules:
        1. Prompt Template  → structured prompt with placeholders
        2. Data Masking     → PII anonymized before any processing
        3. Grounding        → context retrieved from vector store
        4. LLM              → generates answer from context
        5. Content Filter   → output safety check

    This is the recommended pattern for enterprise SAP applications.

    Data flow:
        User Input
          ↓ [Data Masking — PII removed]
          ↓ [Template — filled with masked input]
          ↓ [Grounding — retrieve relevant docs]
          ↓ [LLM — generate answer using context]
          ↓ [Content Filter — safety check]
        Safe Response

    Returns:
        dict with 'answer', 'retrieved_context', 'success'
    """
    # Template must include grounding placeholder
    template = Template(
        messages=[
            SystemMessage(
                "You are a professional HR assistant. "
                "Answer ONLY from the provided context. "
                "Be concise and professional."
            ),
            UserMessage(
                "Context from HR documents:\n"
                "{{ ?grounding_response }}\n\n"
                "Employee Question: {{ ?user_query }}"
            ),
        ]
    )

    # Custom vector store for HR documents
    filters = [
        DocumentGroundingFilter(
            id="hr_documents",
            data_repositories=[vector_repository_id],
            data_repository_type=DataRepositoryType.VECTOR.value,
            search_config={"max_chunk_count": 5}
        )
    ]

    grounding = GroundingModule(
        type=GroundingType.DOCUMENT_GROUNDING_SERVICE.value,
        config=DocumentGrounding(
            input_params=["user_query"],
            output_param="grounding_response",
            filters=filters
        )
    )

    # PII masking — anonymize names/emails before they reach grounding or LLM
    masking = create_data_masking(method="anonymization")

    # Strict content filtering for both input and output
    filtering = create_content_filter(strictness=0)

    # ALL modules combined into one OrchestrationConfig
    config = OrchestrationConfig(
        template=template,
        llm=llm,
        grounding=grounding,
        data_masking=masking,
        filtering=filtering,
    )

    try:
        result = service.run(
            config=config,
            template_values=[
                TemplateValue(name="user_query", value=user_query)
            ]
        )

        answer = result.orchestration_result.choices[0].message.content
        retrieved_context = None
        if result.module_results and result.module_results.grounding:
            retrieved_context = result.module_results.grounding.data.get(
                "grounding_result"
            )

        return {
            "success": True,
            "answer": answer,
            "retrieved_context": retrieved_context
        }

    except Exception as e:
        # Handle content filter violations or other pipeline errors
        return {
            "success": False,
            "answer": None,
            "error": str(e)
        }


# ============================================================
# SECTION 12: CHATBOT WITH CONVERSATION HISTORY
# ============================================================

class SAPChatBot:
    """
    A stateful chatbot that maintains conversation history across turns.

    Why maintain history?
        LLMs are stateless — they don't remember previous messages.
        To have a coherent multi-turn conversation, you must send
        the entire conversation history with every new request.

    How it works:
        Turn 1: [User: "What is SAP BTP?"]
                → send as-is, get response
        Turn 2: [User: "What is SAP BTP?"] + [Asst: "..."] + [User: "Explain AI Core"]
                → send full history, LLM has context of turn 1

    Attributes:
        service:  OrchestrationService instance
        config:   OrchestrationConfig (template + llm)
        history:  List[Message] — growing list of message objects
    """

    def __init__(self, service: OrchestrationService, llm: LLM):
        """
        Initializes the chatbot with a system persona and empty history.

        The template uses {{?user_input}} as the placeholder for each turn.
        History accumulates between calls.
        """
        self.service = service

        # Simple template — one user placeholder per turn
        # History is injected separately via messages_history parameter
        self.config = OrchestrationConfig(
            template=Template(
                messages=[
                    SystemMessage(
                        "You are a knowledgeable SAP consultant. "
                        "Remember previous parts of the conversation."
                    ),
                    UserMessage("{{ ?user_input }}")
                ]
            ),
            llm=llm,
        )

        # history stores all previous messages as Message objects
        # This list grows with each turn and is passed to .run()
        self.history: List = []

    def chat(self, user_input: str) -> str:
        """
        Sends a message and returns the assistant's reply.
        Automatically manages conversation history.

        Args:
            user_input: The user's message for this turn

        Returns:
            The assistant's response string
        """
        # Execute pipeline, passing accumulated history
        # The SDK prepends history messages before the template messages
        result = self.service.run(
            config=self.config,
            messages_history=self.history,   # all previous turns
            template_values=[
                TemplateValue(name="user_input", value=user_input)
            ]
        )

        # Extract the reply message object (not just text)
        # We need the full Message object to add back to history
        reply_message = result.orchestration_result.choices[0].message

        # Append user message to history
        self.history.append(UserMessage(user_input))

        # Append assistant message to history for next turn
        self.history.append(reply_message)

        return reply_message.content

    def reset(self) -> None:
        """Clears conversation history to start fresh."""
        self.history = []
        print("Conversation history cleared.")


# ============================================================
# SECTION 13: LangChain INTEGRATION
# ============================================================

def run_langchain_rag_pipeline(user_query: str) -> str:
    """
    LangChain integration with SAP Gen AI Hub — builds a RAG pipeline
    using FAISS vector store and OpenAI-compatible proxy.

    When to use LangChain vs Orchestration Service?
        Use LangChain:            If you already use LangChain ecosystem
                                  Custom vector stores (Chroma, Weaviate)
                                  Complex agent chains

        Use Orchestration Service: SAP-native solution
                                   Built-in grounding, masking, filtering
                                   Simpler config for standard RAG
                                   Recommended for SAP enterprise apps

    Package versions (correct as of 2024/2025):
        pip install langchain-community langchain-text-splitters faiss-cpu
    """
    # These imports are separate because langchain is optional
    # Install: pip install langchain-community langchain-text-splitters faiss-cpu

    # ChatOpenAI from SAP proxy — routes through SAP AI Core, NOT OpenAI directly
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings

    # Correct 2024+ imports — NOT langchain.vectorstores (deprecated)
    from langchain_community.vectorstores import FAISS

    # Correct 2024+ import — NOT langchain.text_splitter (deprecated)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # LCEL (LangChain Expression Language) components
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # ---- Step 1: Create embedding model ----
    # OpenAIEmbeddings via SAP proxy — converts text → vectors
    # proxy_model_name must match what's deployed in your AI Core
    embeddings = OpenAIEmbeddings(proxy_model_name="text-embedding-ada-002")

    # ---- Step 2: Prepare and chunk documents ----
    documents = [
        "SAP BTP is the Business Technology Platform by SAP.",
        "SAP AI Core is the infrastructure layer for running AI models.",
        "Generative AI Hub provides access to LLMs like GPT-4o and Claude.",
        "Orchestration Service allows combining multiple AI modules in a pipeline.",
    ]

    # RecursiveCharacterTextSplitter: splits long documents into chunks
    # chunk_size:    max characters per chunk
    # chunk_overlap: overlap between chunks to preserve context at boundaries
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents(documents)

    # ---- Step 3: Build FAISS vector store ----
    # FAISS (Facebook AI Similarity Search) — local, fast, no server needed
    # from_documents: embeds each chunk and builds the index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # ---- Step 4: Create retriever ----
    # retriever: fetches top-k similar chunks given a query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ---- Step 5: Create LLM ----
    # ChatOpenAI via SAP proxy — NOT calling OpenAI directly
    llm = ChatOpenAI(
        proxy_model_name="gpt-4o",
        temperature=0.0,       # deterministic for factual RAG
        max_tokens=512
    )

    # ---- Step 6: Build RAG chain with LCEL ----
    # LCEL: LangChain Expression Language — compose chains with | operator
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer only from the provided context.

Context:
{context}

Question: {question}"""
    )

    # Chain composition using pipe operator:
    # 1. {"context": retriever, "question": RunnablePassthrough()}
    #    → retriever fetches context; question passes through unchanged
    # 2. | prompt  → formats into ChatPromptTemplate
    # 3. | llm     → sends to LLM, gets AIMessage back
    # 4. | StrOutputParser() → extracts .content string from AIMessage
    chain = (
        {
            "context": retriever,            # auto-retrieve relevant chunks
            "question": RunnablePassthrough() # pass user query as-is
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execute the full chain
    return chain.invoke(user_query)


# ============================================================
# SECTION 14: MAIN — DEMO RUNNER
# ============================================================

def main():
    """
    Demo runner showing all features.
    Replace the URL and IDs with your actual SAP AI Core values.
    """

    print("=" * 60)
    print("SAP Generative AI Hub SDK — Interview Tutorial Demo")
    print("=" * 60)

    # --- Config ---
    # Replace these with your actual values from SAP AI Launchpad
    ORCHESTRATION_URL = "https://your-orchestration-deployment-url"
    VECTOR_REPO_ID = "your-vector-repository-id"

    # Initialize core objects
    service = create_orchestration_service(ORCHESTRATION_URL)
    llm = create_llm(model_name="gpt-4o")

    print("\n--- 1. Basic Completion ---")
    # Simple one-shot Q&A with no additional modules
    response = run_basic_completion(
        service, llm,
        question="What is SAP AI Core?"
    )
    print(f"Answer: {response}\n")

    print("\n--- 2. Streaming ---")
    # Real-time token streaming — useful for chat UIs
    run_streaming(service, llm, question="Explain SAP Generative AI Hub")

    print("\n--- 3. Grounding with SAP Help Portal ---")
    # RAG using official SAP documentation as knowledge base
    result = run_with_sap_help_grounding(
        service, llm,
        user_query="What are the features of SAP AI Core?"
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources retrieved: {result['retrieved_context']}\n")

    print("\n--- 4. Grounding with Custom Vector Store ---")
    # RAG using your own uploaded documents
    result = run_with_vector_grounding(
        service, llm,
        user_query="What is the refund policy?",
        vector_repository_id=VECTOR_REPO_ID,
        max_chunks=5
    )
    print(f"Answer: {result['answer']}\n")

    print("\n--- 5. Full Enterprise Pipeline (Grounding + Masking + Filtering) ---")
    # Production pipeline with PII protection and content safety
    result = run_full_enterprise_pipeline(
        service, llm,
        user_query="My name is John Smith and I need to know the leave policy",
        vector_repository_id=VECTOR_REPO_ID
    )
    if result["success"]:
        print(f"Answer: {result['answer']}")
    else:
        print(f"Pipeline error: {result['error']}\n")

    print("\n--- 6. Multi-Turn Chatbot ---")
    # Stateful chatbot with automatic history management
    bot = SAPChatBot(service, llm)
    response1 = bot.chat("What is SAP BTP?")
    print(f"Turn 1: {response1}")
    response2 = bot.chat("How does it relate to AI Core?")  # has context of turn 1
    print(f"Turn 2: {response2}")
    bot.reset()

    print("\n--- 7. LangChain RAG (in-memory FAISS) ---")
    # LangChain-based RAG with local vector store
    lc_response = run_langchain_rag_pipeline(
        "What is the Orchestration Service?"
    )
    print(f"LangChain RAG Answer: {lc_response}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
============================================================
INTERVIEW QUICK-REFERENCE CHEAT SHEET
============================================================

Q: What is the new package name for SAP Gen AI SDK?
A: sap-ai-sdk-gen (generative-ai-hub-sdk is deprecated)

Q: What is OrchestrationService?
A: Main client to run/stream AI pipelines deployed on SAP AI Core

Q: What is OrchestrationConfig?
A: Blueprint that bundles template + llm + grounding + masking + filtering

Q: What is the placeholder syntax in templates?
A: {{?variable_name}} — double curly brace + question mark + name

Q: What is grounding / RAG?
A: Retrieve relevant document chunks → inject as context → LLM generates answer

Q: Two types of grounding sources?
A: 1. help.sap.com (SAP official docs), 2. Custom vector store (your documents)

Q: Anonymization vs Pseudonymization?
A: Anon = irreversible replacement, Pseudo = reversible token replacement

Q: Content filter threshold values?
A: 0 = strictest (block all), 6 = most permissive (allow most)

Q: How does multi-turn chatbot work?
A: Pass entire messages_history list to .run() on every turn

Q: What does TemplateValue do?
A: Maps a placeholder name to its runtime value (like a query parameter)

Q: Correct LangChain imports (2024+)?
A: langchain_community.vectorstores (not langchain.vectorstores)
   langchain_text_splitters (not langchain.text_splitter)
   Use LCEL (| operator) instead of deprecated RetrievalQA chain

Q: What is LCEL?
A: LangChain Expression Language — chains components with | pipe operator

Q: result.orchestration_result.choices[0].message.content
A: This is how you extract the text response from an orchestration result

Q: result.module_results.grounding.data['grounding_result']
A: This is how you inspect what was retrieved from the grounding module
"""
