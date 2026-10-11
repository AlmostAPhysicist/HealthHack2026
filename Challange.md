Predict My Glucose Response: Building a Personalized Metabolic Response Function

Challenge
People can have very different glucose responses to the same meal. These differences may depend not only on the food itself, but also on baseline glucose, recent glucose trends, physical activity, heart rate, time of day, prior meals, and diabetes medications.
Your challenge is to build a person-specific glucose response model that learns from seven days of multimodal synthetic data and predicts how that individual will respond to a new meal on Day 8.
Rather than asking only, “How many carbohydrates are in this meal?”, the goal is to answer:
Given what we have learned about this person during the previous seven days, what will happen to their glucose after this meal?

Dataset
Each synthetic participant will have seven days of longitudinal data containing:
•	Continuous glucose monitor data: glucose measurements every 5 minutes
•	Wearable heart rate data: heart rate and resting heart rate, with optional activity-related features
•	Meal information: meal timestamps and photographs of meals
•	Nutrition ground truth: estimated carbohydrate, protein, fat, fiber, and calorie content for training meals
•	Diabetes medication data: medication class, dose, and timing when applicable
•	Time features: time of day and time since prior meal or medication

Participants may have different underlying physiologic characteristics, such as carbohydrate sensitivity, baseline insulin resistance, medication response, morning versus evening glucose responses, or differing effects of recent activity.
On Day 8, teams will receive a new meal photograph together with the individual's recent CGM, heart rate, and medication history. The post-meal glucose response will be hidden.

Task
Develop an algorithm that learns an individualized response function from Days 1–7 and predicts the Day 8 postprandial glucose trajectory.
Teams may use any combination of computer vision, machine learning, time-series modeling, large language models, or physiologically informed approaches.
The model should ideally predict:
•	Glucose at 30, 60, 90, and 120 minutes after the meal
•	Maximum postprandial glucose
•	Change from pre-meal glucose to peak glucose
•	Time to peak glucose
•	Two-hour glucose excursion or incremental area under the curve
Teams are encouraged to go beyond prediction and identify which features appear most important for the individual, such as meal composition, baseline glucose trajectory, heart rate patterns, time of day, or medication timing.
Innovation Opportunity
A strong solution should not simply learn the average response across all participants. It should determine how much of the response is person-specific.
For example, the same meal may produce a large glucose excursion in one participant and a modest response in another because of differences in medication use, baseline glucose, prior activity, or individual carbohydrate sensitivity.
Advanced teams may also estimate uncertainty and identify situations in which the model does not have enough prior information to make a confident prediction.
Deliverable
Teams will provide:
1.	A working prediction model or prototype application
2.	A visualization comparing predicted versus observed Day 8 glucose responses
3.	A short explanation of how the model learns person-specific physiology
4.	Identification of the variables that contributed most to each prediction
5.	A brief discussion of model uncertainty, limitations, and potential clinical applications
Evaluation
Projects will be judged on:
•	Predictive performance: accuracy of the Day 8 glucose prediction
•	Personalization: ability to learn individual rather than population-average responses
•	Interpretability: ability to explain why a prediction was made
•	Innovation: creative integration of meal images, CGM, heart rate, and medication data
•	Usability: potential for translation into a patient-facing or clinician-facing tool

Central Question
Can seven days of multimodal wearable and meal data teach an algorithm enough about an individual to predict how that person will respond to a meal they have not previously eaten?

