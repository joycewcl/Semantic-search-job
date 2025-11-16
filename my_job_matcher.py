import streamlit as st
from backend import JobMatcherBackend
import json

# Page config
st.set_page_config(
    page_title="AI Job Matcher",
    page_icon="🎯",
    layout="wide"
)

# Initialize backend
@st.cache_resource
def load_backend():
    return JobMatcherBackend()

backend = load_backend()

# App UI
st.title("🎯 AI-Powered Job Matcher")
st.markdown("Upload your CV and let **GPT-4** find matching jobs globally, ranked by match quality!")

# Sidebar
st.sidebar.header("📁 Upload Your CV")
cv_file = st.sidebar.file_uploader("Choose your CV", type=['pdf', 'docx'])

# Settings
st.sidebar.header("⚙️ Settings")
num_jobs_to_search = st.sidebar.slider("Jobs to search", 10, 50, 30, 5)
num_jobs_to_show = st.sidebar.slider("Top matches to display", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Note:** Jobs are searched globally and ranked by how well they match your profile, regardless of location.")

if cv_file:
    st.success(f"✅ Uploaded: **{cv_file.name}**")
    
    if st.button("🔍 Analyze with GPT-4 & Find Matching Jobs", type="primary", use_container_width=True):
        
        # STEP 1: Analyze Resume
        with st.spinner("🤖 Step 1/3: Analyzing your resume with GPT-4..."):
            try:
                resume_data, ai_analysis = backend.process_resume(cv_file, cv_file.name)
                
                st.balloons()
                
                # Display AI Analysis
                st.markdown("---")
                st.subheader("🤖 GPT-4 Career Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Primary Role", ai_analysis.get('primary_role', 'N/A'))
                
                with col2:
                    confidence = ai_analysis.get('confidence', 0) * 100
                    st.metric("💯 Confidence", f"{confidence:.0f}%")
                
                with col3:
                    st.metric("📊 Seniority", ai_analysis.get('seniority_level', 'N/A'))
                
                # Skills detected by GPT-4
                st.markdown("### 💡 Skills Detected by GPT-4")
                skills = ai_analysis.get('skills', [])
                if skills:
                    # Create skill tags
                    skills_html = ""
                    for skill in skills[:20]:
                        skills_html += f'<span style="background-color: #E8F4FD; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    if len(skills) > 20:
                        with st.expander(f"➕ Show all {len(skills)} skills"):
                            more_skills_html = ""
                            for skill in skills[20:]:
                                more_skills_html += f'<span style="background-color: #F0F0F0; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                            st.markdown(more_skills_html, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No skills detected")
                
                # Core strengths
                st.markdown("### 💪 Core Strengths")
                strengths = ai_analysis.get('core_strengths', [])
                if strengths:
                    cols = st.columns(len(strengths))
                    for i, strength in enumerate(strengths):
                        with cols[i]:
                            st.info(f"✓ {strength}")
                
                # Job search query
                with st.expander("🔍 AI-Generated Search Query"):
                    st.code(ai_analysis.get('optimal_search_query', 'N/A'), language="text")
                
            except Exception as e:
                st.error(f"❌ Error analyzing resume: {str(e)}")
                st.stop()
        
        # STEP 2: Search & Match Jobs
        with st.spinner(f"🔎 Step 2/3: Searching {num_jobs_to_search} jobs globally via RapidAPI..."):
            try:
                matched_jobs = backend.search_and_match_jobs(
                    resume_data, 
                    ai_analysis, 
                    num_jobs=num_jobs_to_search
                )
            except Exception as e:
                st.error(f"❌ Error searching jobs: {str(e)}")
                st.stop()
        
        # STEP 3: Display Results
        st.markdown("---")
        
        if matched_jobs and len(matched_jobs) > 0:
            st.success(f"✅ Step 3/3: Found & ranked **{len(matched_jobs)}** jobs by match quality!")
            
            st.markdown(f"## 🎯 Top {num_jobs_to_show} Job Matches")
            
            st.info("📊 **Ranking Algorithm:** Combined Score = 60% Semantic Similarity + 40% Skill Match")
            
            # Display top matches
            for i, job in enumerate(matched_jobs[:num_jobs_to_show], 1):
                
                # Determine match quality
                combined = job.get('combined_score', 0)
                if combined >= 80:
                    match_emoji = "🟢"
                    match_label = "Excellent Match"
                    match_color = "#D4EDDA"
                elif combined >= 60:
                    match_emoji = "🟡"
                    match_label = "Good Match"
                    match_color = "#FFF3CD"
                else:
                    match_emoji = "🟠"
                    match_label = "Fair Match"
                    match_color = "#F8D7DA"
                
                # Expander title
                expander_title = f"**#{i}** • {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')} - {match_emoji} {match_label} ({combined:.1f}%)"
                
                with st.expander(expander_title, expanded=(i <= 2)):
                    
                    # Match scores in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🎯 Combined Score", f"{job.get('combined_score', 0):.1f}%")
                    
                    with col2:
                        st.metric("🧠 Semantic Match", f"{job.get('semantic_score', 0):.1f}%")
                    
                    with col3:
                        st.metric("✅ Skill Match", f"{job.get('skill_match_percentage', 0):.1f}%")
                    
                    with col4:
                        st.metric("🔢 Skills Matched", job.get('matched_skills_count', 0))
                    
                    # Job details
                    st.markdown("##### 📋 Job Details")
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.write(f"**📍 Location:** {job.get('location', 'Unknown')}")
                        st.write(f"**🏢 Company:** {job.get('company', 'Unknown')}")
                    
                    with detail_col2:
                        st.write(f"**📅 Posted:** {job.get('posted_date', 'Unknown')}")
                        st.write(f"**💼 Role:** {job.get('title', 'Unknown')}")
                    
                    # Matched skills
                    matched_skills = job.get('matched_skills', [])
                    if matched_skills:
                        st.markdown("##### ✨ Your Skills That Match This Job")
                        skills_matched_html = ""
                        for skill in matched_skills[:8]:
                            skills_matched_html += f'<span style="background-color: #D4EDDA; color: #155724; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block; font-weight: bold;">✓ {skill}</span> '
                        st.markdown(skills_matched_html, unsafe_allow_html=True)
                        
                        if len(matched_skills) > 8:
                            st.caption(f"+ {len(matched_skills) - 8} more matching skills")
                    
                    # Description
                    description = job.get('description', '')
                    if description:
                        st.markdown("##### 📝 Job Description")
                        description_preview = description[:500]
                        st.text_area(
                            "Preview",
                            description_preview + ("..." if len(description) > 500 else ""),
                            height=120,
                            key=f"desc_{job['id']}",
                            disabled=True
                        )
                    
                    # Apply button
                    job_url = job.get('url', '')
                    if job_url:
                        st.link_button(
                            "🔗 Apply Now on LinkedIn",
                            job_url,
                            use_container_width=True,
                            type="primary"
                        )
                    else:
                        st.info("🔗 Application link not available")
            
            # Show all results in table
            st.markdown("---")
            with st.expander("📊 View All Results as Table"):
                import pandas as pd
                
                df_data = []
                for i, job in enumerate(matched_jobs, 1):
                    df_data.append({
                        'Rank': i,
                        'Title': job.get('title', 'N/A'),
                        'Company': job.get('company', 'N/A'),
                        'Location': job.get('location', 'N/A'),
                        'Combined Score': f"{job.get('combined_score', 0):.1f}%",
                        'Semantic': f"{job.get('semantic_score', 0):.1f}%",
                        'Skill Match': f"{job.get('skill_match_percentage', 0):.1f}%",
                        'Skills Matched': job.get('matched_skills_count', 0)
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"job_matches_{cv_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
        
        else:
            st.warning("⚠️ No jobs found. This could be because:")
            st.markdown("""
            - RapidAPI has limited job listings for this role
            - Try adjusting your resume or try again later
            - The search query might be too specific
            """)
        
        # Full GPT-4 Analysis
        st.markdown("---")
        with st.expander("🤖 View Full GPT-4 Analysis (JSON)"):
            st.json(ai_analysis)

else:
    # Welcome screen
    st.info("👈 **Upload your CV from the sidebar to get started!**")
    
    # Instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 How it works:
        
        1. **📄 Upload** your CV (PDF or DOCX)
        2. **🤖 GPT-4** analyzes your skills, experience, and ideal roles
        3. **🔍 Search** LinkedIn jobs via RapidAPI (global search)
        4. **🎯 Rank** all jobs by match quality using AI
        5. **📊 See** your best matches with detailed scores!
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Match Quality Scoring:
        
        **Combined Score** is calculated as:
        - **60%** Semantic Similarity (AI embeddings)
        - **40%** Skill Match Percentage
        
        **Match Levels:**
        - 🟢 **Excellent:** 80%+
        - 🟡 **Good:** 60-79%
        - 🟠 **Fair:** Below 60%
        """)
    
    st.markdown("---")
    st.success("💡 **Pro Tip:** Jobs are searched globally (not filtered by Hong Kong) and ranked by how well they match your profile!")

# Footer
st.markdown("---")
st.caption("🤖 Powered by GPT-4, Pinecone Vector Search, and RapidAPI LinkedIn Jobs")