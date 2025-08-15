"""
AI Fit Scoring Tool - Authentication Module
Handles Supabase authentication and user management
"""

import streamlit as st
import os
from datetime import datetime, timedelta
import json
from typing import Dict, Optional, List, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseAuth:
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            st.error("⚠️ Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in your environment variables.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                st.error(f"Failed to initialize Supabase client: {str(e)}")
                self.client = None
    
    def is_initialized(self) -> bool:
        """Check if Supabase client is properly initialized"""
        return self.client is not None
    
    def sign_up(self, email: str, password: str, full_name: str = "") -> Tuple[bool, str]:
        """Sign up a new user"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            # Create user with metadata
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name or email.split("@")[0]
                    }
                }
            })
            
            if response.user:
                return True, "Account created successfully! Please check your email for verification."
            else:
                return False, "Failed to create account. Please try again."
                
        except Exception as e:
            error_msg = str(e)
            if "already registered" in error_msg.lower():
                return False, "An account with this email already exists."
            elif "invalid email" in error_msg.lower():
                return False, "Please enter a valid email address."
            elif "password" in error_msg.lower():
                return False, "Password must be at least 6 characters long."
            else:
                return False, f"Sign up failed: {error_msg}"
    
    def sign_in(self, email: str, password: str) -> Tuple[bool, str]:
        """Sign in an existing user"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                # Store user session in Streamlit session state
                st.session_state.user = response.user
                st.session_state.session = response.session
                
                # Log usage analytics
                self.log_user_action("login")
                
                return True, "Successfully signed in!"
            else:
                return False, "Invalid email or password."
                
        except Exception as e:
            error_msg = str(e)
            if "invalid" in error_msg.lower():
                return False, "Invalid email or password."
            elif "too many" in error_msg.lower():
                return False, "Too many login attempts. Please try again later."
            else:
                return False, f"Sign in failed: {error_msg}"
    
    def sign_out(self) -> bool:
        """Sign out the current user"""
        if not self.is_initialized():
            return False
        
        try:
            self.client.auth.sign_out()
            
            # Clear Streamlit session state
            if 'user' in st.session_state:
                del st.session_state.user
            if 'session' in st.session_state:
                del st.session_state.session
            if 'user_profile' in st.session_state:
                del st.session_state.user_profile
            
            return True
            
        except Exception as e:
            st.error(f"Sign out failed: {str(e)}")
            return False
    
    def get_current_user(self) -> Optional[Dict]:
        """Get the current authenticated user"""
        if not self.is_initialized():
            return None
        
        try:
            # Check if user is in session state
            if 'user' in st.session_state and st.session_state.user:
                return st.session_state.user
            
            # Try to get user from Supabase session
            user = self.client.auth.get_user()
            if user and user.user:
                st.session_state.user = user.user
                return user.user
            
            return None
            
        except Exception:
            return None
    
    def get_user_profile(self, user_id: str = None) -> Optional[Dict]:
        """Get user profile from database"""
        if not self.is_initialized():
            return None
        
        try:
            current_user = self.get_current_user()
            if not current_user and not user_id:
                return None
            
            user_id = user_id or current_user.id
            
            response = self.client.table("user_profiles").select("*").eq("id", user_id).execute()
            
            if response.data and len(response.data) > 0:
                profile = response.data[0]
                st.session_state.user_profile = profile
                return profile
            
            return None
            
        except Exception as e:
            st.error(f"Failed to load user profile: {str(e)}")
            return None
    
    def update_user_profile(self, profile_data: Dict) -> Tuple[bool, str]:
        """Update user profile"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return False, "User not authenticated"
            
            # Add updated_at timestamp
            profile_data['updated_at'] = datetime.now().isoformat()
            
            response = self.client.table("user_profiles").update(profile_data).eq("id", current_user.id).execute()
            
            if response.data:
                # Update session state
                st.session_state.user_profile = response.data[0]
                return True, "Profile updated successfully!"
            else:
                return False, "Failed to update profile"
                
        except Exception as e:
            return False, f"Update failed: {str(e)}"
    
    def save_project(self, project_data: Dict) -> Tuple[bool, str, Optional[str]]:
        """Save a project to the database"""
        if not self.is_initialized():
            return False, "Authentication service not available", None
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return False, "User not authenticated", None
            
            # Add user_id and timestamps
            project_data['user_id'] = current_user.id
            project_data['created_at'] = datetime.now().isoformat()
            project_data['updated_at'] = datetime.now().isoformat()
            
            response = self.client.table("projects").insert(project_data).execute()
            
            if response.data and len(response.data) > 0:
                project_id = response.data[0]['id']
                
                # Log usage analytics
                self.log_user_action("project_created", {"project_id": project_id})
                
                return True, "Project saved successfully!", project_id
            else:
                return False, "Failed to save project", None
                
        except Exception as e:
            return False, f"Save failed: {str(e)}", None
    
    def save_scoring_results(self, project_id: str, scoring_data: Dict, nlp_analysis: Dict = None) -> Tuple[bool, str]:
        """Save scoring results to the database"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return False, "User not authenticated"
            
            # Prepare scoring data
            scoring_record = {
                'project_id': project_id,
                'user_id': current_user.id,
                'overall_score': scoring_data.get('overall_score', 0),
                'data_requirements_score': scoring_data.get('data_requirements_score', 3),
                'data_requirements_reasoning': scoring_data.get('data_requirements_reasoning', ''),
                'problem_complexity_score': scoring_data.get('problem_complexity_score', 3),
                'problem_complexity_reasoning': scoring_data.get('problem_complexity_reasoning', ''),
                'business_impact_score': scoring_data.get('business_impact_score', 3),
                'business_impact_reasoning': scoring_data.get('business_impact_reasoning', ''),
                'technical_feasibility_score': scoring_data.get('technical_feasibility_score', 3),
                'technical_feasibility_reasoning': scoring_data.get('technical_feasibility_reasoning', ''),
                'timeline_alignment_score': scoring_data.get('timeline_alignment_score', 3),
                'timeline_alignment_reasoning': scoring_data.get('timeline_alignment_reasoning', ''),
                'nlp_analysis': json.dumps(nlp_analysis) if nlp_analysis else None,
                'scoring_method': scoring_data.get('scoring_method', 'manual'),
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table("scoring_results").insert(scoring_record).execute()
            
            if response.data:
                # Log usage analytics
                self.log_user_action("scoring_completed", {"project_id": project_id, "method": scoring_record['scoring_method']})
                
                return True, "Scoring results saved successfully!"
            else:
                return False, "Failed to save scoring results"
                
        except Exception as e:
            return False, f"Save failed: {str(e)}"
    
    def get_user_projects(self, limit: int = 50) -> List[Dict]:
        """Get user's projects from the database"""
        if not self.is_initialized():
            return []
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return []
            
            response = self.client.table("projects").select("*").eq("user_id", current_user.id).order("created_at", desc=True).limit(limit).execute()
            
            return response.data or []
            
        except Exception as e:
            st.error(f"Failed to load projects: {str(e)}")
            return []
    
    def get_project_scoring_history(self, project_id: str) -> List[Dict]:
        """Get scoring history for a specific project"""
        if not self.is_initialized():
            return []
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return []
            
            response = self.client.table("scoring_results").select("*").eq("project_id", project_id).eq("user_id", current_user.id).order("created_at", desc=True).execute()
            
            return response.data or []
            
        except Exception as e:
            st.error(f"Failed to load scoring history: {str(e)}")
            return []
    
    def delete_project(self, project_id: str) -> Tuple[bool, str]:
        """Delete a project and its associated scoring results"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return False, "User not authenticated"
            
            # Delete scoring results first (due to foreign key constraint)
            self.client.table("scoring_results").delete().eq("project_id", project_id).eq("user_id", current_user.id).execute()
            
            # Delete project
            response = self.client.table("projects").delete().eq("id", project_id).eq("user_id", current_user.id).execute()
            
            if response.data is not None:  # Supabase returns None for successful deletes
                return True, "Project deleted successfully!"
            else:
                return False, "Failed to delete project"
                
        except Exception as e:
            return False, f"Delete failed: {str(e)}"
    
    def submit_feedback(self, feedback_data: Dict) -> Tuple[bool, str]:
        """Submit user feedback"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return False, "User not authenticated"
            
            feedback_record = {
                'user_id': current_user.id,
                'feedback_type': feedback_data.get('feedback_type', 'general'),
                'rating': feedback_data.get('rating'),
                'feedback_text': feedback_data.get('feedback_text', ''),
                'project_id': feedback_data.get('project_id'),
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table("feedback").insert(feedback_record).execute()
            
            if response.data:
                return True, "Feedback submitted successfully! Thank you for helping us improve."
            else:
                return False, "Failed to submit feedback"
                
        except Exception as e:
            return False, f"Submission failed: {str(e)}"
    
    def log_user_action(self, action_type: str, metadata: Dict = None):
        """Log user action for analytics"""
        if not self.is_initialized():
            return
        
        try:
            current_user = self.get_current_user()
            if not current_user:
                return
            
            analytics_record = {
                'user_id': current_user.id,
                'action_type': action_type,
                'metadata': json.dumps(metadata) if metadata else None,
                'created_at': datetime.now().isoformat()
            }
            
            # Don't wait for response, fire and forget
            self.client.table("usage_analytics").insert(analytics_record).execute()
            
        except Exception:
            # Silently fail analytics logging to not disrupt user experience
            pass
    
    def reset_password(self, email: str) -> Tuple[bool, str]:
        """Send password reset email"""
        if not self.is_initialized():
            return False, "Authentication service not available"
        
        try:
            self.client.auth.reset_password_email(email)
            return True, "Password reset email sent! Please check your inbox."
            
        except Exception as e:
            return False, f"Reset failed: {str(e)}"

# Global authentication instance
auth_manager = SupabaseAuth()

def require_auth(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        current_user = auth_manager.get_current_user()
        if not current_user:
            st.warning("🔒 Please sign in to access this feature.")
            return None
        return func(*args, **kwargs)
    return wrapper

def get_user_display_name() -> str:
    """Get user's display name for UI"""
    current_user = auth_manager.get_current_user()
    if not current_user:
        return "Guest"
    
    profile = auth_manager.get_user_profile()
    if profile and profile.get('full_name'):
        return profile['full_name']
    
    return current_user.email.split('@')[0] if current_user.email else "User"

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return auth_manager.get_current_user() is not None