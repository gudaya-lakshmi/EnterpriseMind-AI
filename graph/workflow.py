from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END


# ============================================================
# CONFIGURATION
# ============================================================

MAX_VERIFICATION_RETRIES = 2


# ============================================================
# LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):

    question: str
        # Security Agent
    security_status: str
    security_reason: str
    security_allowed: bool
    security_risk_score: int

    # Router Agent
    route: str
    route_reason: str

    # Retrieval
    retrieval_query: str
    candidate_documents: list
    documents: list
    context: str

    # Summary Scope Filter
    summary_documents: list
    summary_context: str

    # Answer / Summary
    answer: str

    # Verification
    verification: dict[str, Any]
    retry_count: int

    # Citation Agent
    citation_result: dict[str, Any]
    citations: list

    # Final output
    final_answer: str


# ============================================================
# BUILD WORKFLOW
# ============================================================

def build_rag_workflow(
    retriever,
    rerank_documents,
    create_retrieval_query,
    build_context,
    llm,
    prompt_template,
    security_agent,
    security_rejection_node,
    route_after_security,
    verify_answer,
    find_citations,
    route_request,
    summarize_documents,
    filter_summary_documents,
):


    # ========================================================
    # ROUTER AGENT NODE
    # ========================================================

    def router_node(state: RAGState) -> dict:

        print(
            "\n[LangGraph] Router Agent analyzing request..."
        )

        question = state["question"]

        routing_result = route_request(
            question
        )

        route = routing_result.get(
            "route",
            "qa"
        )

        reason = routing_result.get(
            "reason",
            ""
        )

        if route not in {
            "qa",
            "summarize"
        }:

            route = "qa"

            reason = (
                "Invalid route returned. "
                "Defaulting to QA."
            )

        print(
            "[LangGraph] Router decision:",
            route
        )

        if reason:

            print(
                "[LangGraph] Router reason:",
                reason
            )

        return {
            "route": route,
            "route_reason": reason,
            "retry_count": 0,
        }


    # ========================================================
    # RETRIEVAL NODE
    # ========================================================

    def retrieve_node(state: RAGState) -> dict:

        print(
            "[LangGraph] Retrieving documents..."
        )

        question = state["question"]

        retrieval_query = create_retrieval_query(
            question
        )

        candidate_documents = retriever.invoke(
            retrieval_query
        )

        print(
            f"[LangGraph] Retrieved "
            f"{len(candidate_documents)} candidate chunks."
        )

        return {
            "retrieval_query": retrieval_query,
            "candidate_documents": candidate_documents,
        }


    # ========================================================
    # RERANK NODE
    # ========================================================

    def rerank_node(state: RAGState) -> dict:

        print(
            "[LangGraph] Reranking documents..."
        )

        question = state["question"]

        candidate_documents = state.get(
            "candidate_documents",
            []
        )

        if not candidate_documents:

            return {
                "documents": [],
                "context": "",
            }

        documents = rerank_documents(
            question,
            candidate_documents
        )

        print(
            f"[LangGraph] After reranking: "
            f"{len(documents)} context chunks."
        )

        context = build_context(
            documents
        )

        return {
            "documents": documents,
            "context": context,
        }


    # ========================================================
    # ROUTE AFTER RERANKING
    # ========================================================

    def route_after_reranking(
        state: RAGState
    ) -> str:

        route = state.get(
            "route",
            "qa"
        )

        if route == "summarize":

            print(
                "[LangGraph] Sending evidence "
                "to Summary Scope Filter."
            )

            return "summary_filter"

        print(
            "[LangGraph] Sending evidence "
            "to QA Generator."
        )

        return "generate"


    # ========================================================
    # SUMMARY SCOPE FILTER NODE
    # ========================================================

    def summary_filter_node(
        state: RAGState
    ) -> dict:

        print(
            "[LangGraph] Filtering evidence "
            "for summary scope..."
        )

        question = state["question"]

        documents = state.get(
            "documents",
            []
        )

        if not documents:

            return {
                "summary_documents": [],
                "summary_context": "",
            }

        filtered_documents = filter_summary_documents(
            question=question,
            documents=documents,
        )

        print(
            f"[LangGraph] Summary scope filter kept "
            f"{len(filtered_documents)} of "
            f"{len(documents)} chunks."
        )

        if not filtered_documents:

            return {
                "summary_documents": [],
                "summary_context": "",
            }

        summary_context = build_context(
            filtered_documents
        )

        return {
            "summary_documents": filtered_documents,
            "summary_context": summary_context,
        }


    # ========================================================
    # NORMAL QA GENERATOR NODE
    # ========================================================

    def generate_node(state: RAGState) -> dict:

        print(
            "[LangGraph] Generating QA answer..."
        )

        question = state["question"]

        context = state.get(
            "context",
            ""
        )

        if not context.strip():

            return {
                "answer": (
                    "I could not find this information "
                    "in the provided documents."
                )
            }

        prompt = prompt_template.format(
            context=context,
            question=question,
        )

        response = llm.invoke(
            prompt
        )

        answer = getattr(
            response,
            "content",
            str(response)
        ).strip()

        if not answer:

            answer = (
                "I could not find this information "
                "in the provided documents."
            )

        return {
            "answer": answer
        }


    # ========================================================
    # SUMMARIZER AGENT NODE
    # ========================================================

    def summarize_node(
        state: RAGState
    ) -> dict:

        print(
            "[LangGraph] Summarizer Agent "
            "generating summary..."
        )

        question = state["question"]

        context = state.get(
            "summary_context",
            ""
        )

        if not context.strip():

            return {
                "answer": (
                    "I could not find enough information "
                    "in the provided documents to summarize "
                    "this topic."
                )
            }

        result = summarize_documents(
            question=question,
            context=context,
        )

        summary = result.get(
            "summary",
            ""
        )

        if not summary:

            summary = (
                "I could not find enough information "
                "in the provided documents to summarize "
                "this topic."
            )

        return {
            "answer": summary
        }


    # ========================================================
    # VERIFIER AGENT NODE
    # ========================================================

    def verify_node(
        state: RAGState
    ) -> dict:

        print(
            "[LangGraph] Verifying answer..."
        )

        route = state.get(
            "route",
            "qa"
        )

        if route == "summarize":

            verification_context = state.get(
                "summary_context",
                ""
            )

        else:

            verification_context = state.get(
                "context",
                ""
            )

        verification = verify_answer(
            question=state["question"],
            answer=state.get(
                "answer",
                ""
            ),
            context=verification_context,
        )

        print(
            "[LangGraph] Verifier verdict:",
            verification.get(
                "verdict",
                "UNKNOWN"
            )
        )

        issues = verification.get(
            "issues",
            []
        )

        if issues:

            print(
                "[LangGraph] Verifier issues:",
                issues
            )

        return {
            "verification": verification
        }


    # ========================================================
    # ROUTING AFTER VERIFICATION
    # ========================================================

    def route_after_verification(
        state: RAGState
    ) -> str:

        verification = state.get(
            "verification",
            {}
        )

        verdict = verification.get(
            "verdict",
            "REVISE"
        )

        retry_count = state.get(
            "retry_count",
            0
        )

        if verdict == "PASS":

            print(
                "[LangGraph] Answer accepted."
            )

            return "citation"

        if (
            retry_count
            >= MAX_VERIFICATION_RETRIES
        ):

            print(
                "[LangGraph] Maximum revision "
                "attempts reached."
            )

            return "citation"

        print(
            "[LangGraph] Sending answer "
            "for revision."
        )

        return "revise"


    # ========================================================
    # REVISION NODE
    # ========================================================

    def revise_node(
        state: RAGState
    ) -> dict:

        retry_count = (
            state.get(
                "retry_count",
                0
            )
            + 1
        )

        route = state.get(
            "route",
            "qa"
        )

        print(
            f"[LangGraph] Revising "
            f"{route} response "
            f"({retry_count}/"
            f"{MAX_VERIFICATION_RETRIES})..."
        )

        question = state["question"]

        answer = state.get(
            "answer",
            ""
        )

        verification = state.get(
            "verification",
            {}
        )

        feedback = verification.get(
            "feedback",
            ""
        )

        issues = verification.get(
            "issues",
            []
        )


        # ----------------------------------------------------
        # SUMMARIZATION REVISION
        # ----------------------------------------------------

        if route == "summarize":

            context = state.get(
                "summary_context",
                ""
            )

            revision_prompt = f"""
You are revising a grounded enterprise document summary.

Use ONLY the filtered document evidence below.

User request:
{question}

Previous summary:
{answer}

Verifier feedback:
{feedback}

Verifier issues:
{issues}

Filtered document evidence:
{context}

Instructions:

1. Correct every issue identified by the verifier.

2. Stay strictly within the topic and scope requested by the user.

3. Use ONLY information explicitly supported by the filtered evidence.

4. Do not use outside knowledge.

5. Do not invent facts, numbers, percentages, dates,
   explanations, causes, or conclusions.

6. Preserve important financial category names exactly.

7. Preserve exact numbers, percentages, dates,
   and monetary values.

8. Preserve increase and decrease directions exactly.

9. Preserve important cause-and-effect relationships.

10. Remove irrelevant information.

11. Remove repetition.

12. Do not mention the verification process.

13. Do not mention evidence, context, chunks,
    filtering, or internal agents.

14. If there is genuinely not enough information,
    respond exactly:

"I could not find enough information in the provided documents to summarize this topic."

Revised summary:
"""


        # ----------------------------------------------------
        # QA REVISION
        # ----------------------------------------------------

        else:

            context = state.get(
                "context",
                ""
            )

            revision_prompt = f"""
You are revising an Enterprise Financial Document answer.

Use ONLY the retrieved evidence below.

Original question:
{question}

Previous answer:
{answer}

Verifier feedback:
{feedback}

Verifier issues:
{issues}

Retrieved evidence:
{context}

Instructions:

1. Correct every issue identified by the verifier.

2. Use only information explicitly supported by the retrieved evidence.

3. Preserve exact financial category names.

4. Preserve exact numbers, percentages, dates,
   and monetary values.

5. Do not introduce outside knowledge.

6. Do not invent information.

7. Do not reverse an increase into a decrease
   or a decrease into an increase.

8. Preserve cause-and-effect relationships exactly
   as stated in the evidence.

9. Do not state that figures, percentages,
   or information are unavailable when they
   appear in the evidence.

10. Answer the user's exact question directly.

11. Do not mention the verification process.

12. Do not mention retrieved evidence or context
    in the final answer.

13. If the evidence genuinely does not contain
    the answer, say exactly:

"I could not find this information in the provided documents."

Revised answer:
"""

        response = llm.invoke(
            revision_prompt
        )

        revised_answer = getattr(
            response,
            "content",
            str(response)
        ).strip()

        if not revised_answer:

            revised_answer = answer

        return {
            "answer": revised_answer,
            "retry_count": retry_count,
        }


    # ========================================================
    # CITATION AGENT NODE
    # ========================================================

    def citation_node(
        state: RAGState
    ) -> dict:

        print(
            "[LangGraph] Finding supporting citations..."
        )

        question = state["question"]

        answer = state.get(
            "answer",
            ""
        )

        route = state.get(
            "route",
            "qa"
        )

        if route == "summarize":

            documents = state.get(
                "summary_documents",
                []
            )

        else:

            documents = state.get(
                "documents",
                []
            )

        if not documents:

            print(
                "[LangGraph] No documents available "
                "for citation."
            )

            return {
                "citation_result": {
                    "supported": False,
                    "citations": [],
                },
                "citations": [],
            }

        citation_result = find_citations(
            question=question,
            answer=answer,
            documents=documents,
        )

        citations = citation_result.get(
            "citations",
            []
        )

        print(
            f"[LangGraph] Citation Agent selected "
            f"{len(citations)} supporting source(s)."
        )

        for citation in citations:

            print(
                f"[LangGraph] Citation: "
                f"{citation.get('source')} "
                f"— Page "
                f"{citation.get('page')}"
            )

        return {
            "citation_result": citation_result,
            "citations": citations,
        }


    # ========================================================
    # FINALIZE NODE
    # ========================================================

    def finalize_node(
        state: RAGState
    ) -> dict:

        print(
            "[LangGraph] Finalizing answer..."
        )

        answer = state.get(
            "answer",
            ""
        )

        route = state.get(
            "route",
            "qa"
        )

        if not answer:

            if route == "summarize":

                answer = (
                    "I could not find enough information "
                    "in the provided documents to summarize "
                    "this topic."
                )

            else:

                answer = (
                    "I could not find this information "
                    "in the provided documents."
                )

        return {
            "final_answer": answer
        }


    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    workflow = StateGraph(
        RAGState
    )


    # ========================================================
    # ADD NODES
    # ========================================================

    # Security Agent
    workflow.add_node(
        "security",
        security_agent
    )

    workflow.add_node(
        "security_rejection",
        security_rejection_node
    )


    # Router Agent
    workflow.add_node(
        "router",
        router_node
    )

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "rerank",
        rerank_node
    )

    workflow.add_node(
        "summary_filter",
        summary_filter_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_node(
        "summarize",
        summarize_node
    )

    workflow.add_node(
        "verify",
        verify_node
    )

    workflow.add_node(
        "revise",
        revise_node
    )

    workflow.add_node(
        "citation",
        citation_node
    )

    workflow.add_node(
        "finalize",
        finalize_node
    )


        # ========================================================
    # START → SECURITY
    # ========================================================

    workflow.add_edge(
        START,
        "security"
    )


    # ========================================================
    # SECURITY → ROUTER OR REJECTION
    # ========================================================

    workflow.add_conditional_edges(
        "security",
        route_after_security,
        {
            "allowed": "router",
            "blocked": "security_rejection",
        },
    )


    # ========================================================
    # SECURITY REJECTION → END
    # ========================================================

    workflow.add_edge(
        "security_rejection",
        END
    )

    # ========================================================
    # ROUTER → RETRIEVAL
    # ========================================================

    workflow.add_edge(
        "router",
        "retrieve"
    )


    # ========================================================
    # RETRIEVAL → RERANK
    # ========================================================

    workflow.add_edge(
        "retrieve",
        "rerank"
    )


    # ========================================================
    # RERANK → QA OR SUMMARY FILTER
    # ========================================================

    workflow.add_conditional_edges(
        "rerank",
        route_after_reranking,
        {
            "generate": "generate",
            "summary_filter": "summary_filter",
        },
    )


    # ========================================================
    # SUMMARY FILTER → SUMMARIZER
    # ========================================================

    workflow.add_edge(
        "summary_filter",
        "summarize"
    )


    # ========================================================
    # QA GENERATOR → VERIFIER
    # ========================================================

    workflow.add_edge(
        "generate",
        "verify"
    )


    # ========================================================
    # SUMMARIZER → VERIFIER
    # ========================================================

    workflow.add_edge(
        "summarize",
        "verify"
    )


    # ========================================================
    # VERIFIER → REVISION OR CITATION
    # ========================================================

    workflow.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "revise": "revise",
            "citation": "citation",
        },
    )


    # ========================================================
    # REVISION → VERIFIER
    # ========================================================

    workflow.add_edge(
        "revise",
        "verify"
    )


    # ========================================================
    # CITATION → FINALIZE
    # ========================================================

    workflow.add_edge(
        "citation",
        "finalize"
    )


    # ========================================================
    # FINALIZE → END
    # ========================================================

    workflow.add_edge(
        "finalize",
        END
    )


    return workflow.compile()