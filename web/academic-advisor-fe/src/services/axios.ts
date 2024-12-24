import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

const BE_BASE_URL: string = 'http://localhost:8080';


const response: AxiosInstance = axios.create({
  baseURL: BE_BASE_URL, // URL cho localhost
});


response.interceptors.response.use(
  function (response: AxiosResponse) {
    
    return response.data;
  },
  function (error: AxiosError) {
    
    return Promise.reject(error);
  },
);

export { response, BE_BASE_URL };
