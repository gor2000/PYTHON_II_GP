import streamlit as st


def main():
    st.set_page_config(
        page_title="Bike Sharing Business Strategies",
        layout="wide",
        page_icon="🚲"
    )

    st.markdown("<h1 style='text-align: center;'>🚲 Bike Sharing Business Strategies</h1>", unsafe_allow_html=True)

    st.markdown("---")
    strategies = {
        "🏆 Rewards System": {
            "Objective": "Encourage user loyalty and promote off-peak usage, exercise, and wellness.",
            "Implementation": "Introduce a rewards system where users accumulate riding kilometers. They can redeem these kilometers for prizes like free bicycle time. This will incentivize users to ride more and choose off-peak hours, contributing to their well-being and a more sustainable commuting system."
        },
        "⛈️ Double Kms in Poor Weather Conditions": {
            "Objective": "Boost usage during adverse weather conditions.",
            "Implementation": "Offer double kilometers for rides during rainy days or poor weather conditions. This strategy not only encourages usage in less favorable weather but also enhances user satisfaction by providing extra value during challenging times."
        },
        "☔ Rainy Day Bicycle Plastic Roofs": {
            "Objective": "Enhance user experience during rainy days.",
            "Implementation": "Provide an optional rental service for plastic bicycle roofs on rainy days. Given that for obvious reason we see a decline of 45% average hourly rentals due to poor weather conditions, this will be an important issue to try to address. This ensures users can continue riding comfortably despite the weather, fostering a positive and reliable service image.",
            "Images":["Images/Bub-up-2.jpg", "Images/Roofbi_4.jpg"]
        },
        "💥 Dynamic Pricing": {
            "Objective": "Optimize pricing based on demand fluctuations.",
            "Implementation": "Implement dynamic pricing, adjusting rates based on demand during different hours. Higher prices during peak hours incentivize off-peak usage, balancing demand and optimizing resource allocation."
        },
        "🤖 Machine Learning Predictive Model": {
            "Objective": "Efficient resource allocation and maintenance scheduling.",
            "Implementation": "Utilize a machine learning predictive model for demand forecasting. This allows proactive resource allocation, ensuring an optimal number of bikes at each station. Additionally, it aids in planning maintenance schedules, reducing downtime."
        },
        "🤳🏻 Social Media Influencer Marketing": {
            "Objective": "Increase brand visibility and attract new users.",
            "Implementation": "Collaborate with TikTok and Instagram influencers to promote city day plans on bikes. Engage influencers to share their positive experiences, showcasing the fun and convenience of using bicycles for daily activities."
        },
        "🅿 Optimized Parking Suggestions": {
            "Objective": "Enhance user experience by addressing parking challenges.",
            "Implementation": "Use machine learning predictions to suggest optimal parking spots based on users' destinations. This feature minimizes the inconvenience of finding parking spots, providing a seamless experience for users."
        },
        "😎 Tourist Promotions": {
            "Objective": "Attract and retain tourist users.",
            "Implementation": "Launch marketing campaigns and promotions targeting tourists. Simplify the onboarding process for tourists, making it attractive for them to sign up. Special promotions can include discounted rates or bonus kilometers for their first rides."
        },
        "🧋 Collaboration with Local Businesses": {
            "Objective": "Foster community engagement and support local businesses.",
            "Implementation": "Collaborate with local businesses to include them in the rewards program. Users reaching milestones can receive rewards like free breakfast or coffee at partnering local shops. This builds a sense of community and promotes local businesses."
        },
        "💼 Corporate Partnerships": {
            "Objective": "Encourage sustainable commuting for employees.",
            "Implementation": "Partner with offices and companies to offer special rates for employees' commutes. Provide incentives for employees to choose a sustainable mode of transportation, contributing to a greener and healthier workplace."
        }
    }

    for strategy_title, strategy_details in strategies.items():
        with st.expander(f"#### {strategy_title}"):
            st.write(f"**Objective:** {strategy_details['Objective']}")
            st.write(f"**Implementation:** {strategy_details['Implementation']}")
            
            # Check if there are images for the current strategy
            if 'Images' in strategy_details and len(strategy_details['Images']) == 2:
                col1, col2 = st.columns(2)
                col1.image(strategy_details['Images'][0], caption=f"{strategy_title}", use_column_width=True)
                col2.image(strategy_details['Images'][1], caption=f"{strategy_title}", use_column_width=True)

            st.markdown("---")

if __name__ == "__main__":
    main()



hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
 